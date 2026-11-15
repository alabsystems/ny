// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// exec_shim.c -- LD_PRELOAD exec interposer for the release-candidate smoke rig
// (scripts/release/smoke_submission.sh). It is test equipment: it is never
// linked into ny and never shipped in a submission.
//
// WHY. The smoke rig runs the OFFICIAL VNN-COMP contract -- install_tool.sh,
// then the organizers' run_single_instance.sh, which runs prepare_instance.sh
// and run_instance.sh -- on this aarch64 host against the packaged x86_64
// binary. Registering a binfmt_misc handler needs root (and unprivileged user
// namespaces are AppArmor-restricted here), so the kernel refuses to exec an
// x86_64 ELF with ENOEXEC. This shim is a process-scoped binfmt_misc: every
// exec*/posix_spawn* that names an x86_64 ELF is rewritten to
//
//     <qemu-x86_64> -cpu <cpu> -L <sysroot> -0 <argv0> <path> <argv1..>
//
// and nothing else is changed. It can also report `x86_64` as uname(2)'s
// machine, so the harness scripts take their x86_64 branches exactly as they
// would on the evaluation host (qemu-user does the same for the guest).
//
// EVIDENCE. Every exec is logged, one line each, with the NAMES of the NY_* and
// AY_* variables the new process receives. That is how the rig proves that
// run_instance.sh hands the verifier no NY_* variable (and that the caller's
// canary really reached the contract's entry point, so the absence is not an
// empty grep).
//
// When an x86_64 ELF is routed, LD_PRELOAD and every X86SHIM_* variable are
// removed from the environment the binary receives: the verifier sees exactly
// the environment the harness gave it, minus the rig's own control variables.
//
// LIMITATION (stated, not hidden). qemu-user executes the guest's own execve
// natively, so ny's self-exec children (the ONNX Runtime shape-inference child,
// the external watchdog) still fail with ENOEXEC under this rig and ny takes
// its in-process fallbacks. A real x86_64 host does not.
//
// Control variables, read from the CALLING process:
//   X86SHIM_LOG              append-only exec log (absent: no logging)
//   X86SHIM_QEMU             qemu-x86_64 (absent: nothing is routed)
//   X86SHIM_QEMU_CPU         CPU model (default Haswell-v4)
//   X86SHIM_QEMU_SYSROOT     x86_64 runtime sysroot for -L
//   X86SHIM_UNAME_MACHINE    machine string uname(2) reports (absent: unchanged)
//   X86SHIM_SUBJECT          the binary under test: when it is exec'd natively
//                            (no routing), the rig's variables are still removed
//                            from its environment, exactly as for a routed exec
//
// No allocation: every buffer is on the stack, because the exec wrappers run in
// a forked child where malloc is not safe.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

extern char **environ;

#define SHIM_MAX_ARGS 4096
#define SHIM_MAX_ENV 8192

typedef int (*execve_fn)(const char *, char *const[], char *const[]);
typedef int (*spawn_fn)(pid_t *, const char *, const posix_spawn_file_actions_t *,
                        const posix_spawnattr_t *, char *const[], char *const[]);
typedef int (*uname_fn)(struct utsname *);

static execve_fn real_execve(void) {
    static execve_fn fn;
    if (!fn) {
        fn = (execve_fn)dlsym(RTLD_NEXT, "execve");
    }
    return fn;
}

static spawn_fn real_spawn(const char *name) {
    return (spawn_fn)dlsym(RTLD_NEXT, name);
}

// e_machine of an ELF64 little-endian file, or -1.
static int elf64_machine(const char *path) {
    unsigned char header[20];
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    ssize_t got = read(fd, header, sizeof header);
    close(fd);
    if (got != (ssize_t)sizeof header || memcmp(header, "\177ELF", 4) != 0 || header[4] != 2
        || header[5] != 1) {
        return -1;
    }
    return header[18] | (header[19] << 8);
}

static int routes(const char *path) {
    const char *qemu = getenv("X86SHIM_QEMU");
    return qemu != NULL && qemu[0] != '\0' && path != NULL && elf64_machine(path) == 62;
}

// The binary under test, run natively: same environment hygiene as a routed
// exec, so the shim is never loaded into the verifier itself.
static int is_subject(const char *path) {
    const char *subject = getenv("X86SHIM_SUBJECT");
    char want[PATH_MAX];
    char got[PATH_MAX];
    if (subject == NULL || subject[0] == '\0' || path == NULL) {
        return 0;
    }
    if (realpath(subject, want) == NULL || realpath(path, got) == NULL) {
        return 0;
    }
    return strcmp(want, got) == 0;
}

static int drops_from_guest(const char *entry) {
    return strncmp(entry, "LD_PRELOAD=", 11) == 0 || strncmp(entry, "X86SHIM_", 8) == 0;
}

// Copy envp into out, dropping the rig's variables. Returns the count.
static size_t guest_env(char *const envp[], char **out, size_t cap) {
    size_t n = 0;
    for (size_t i = 0; envp != NULL && envp[i] != NULL && n + 1 < cap; i++) {
        if (!drops_from_guest(envp[i])) {
            out[n++] = envp[i];
        }
    }
    out[n] = NULL;
    return n;
}

struct line {
    char text[8192];
    size_t len;
};

static void put(struct line *line, const char *text) {
    while (text != NULL && *text != '\0' && line->len + 2 < sizeof line->text) {
        char c = *text++;
        line->text[line->len++] = (c == '\n' || c == '\t') ? ' ' : c;
    }
}

static void put_number(struct line *line, long value) {
    char digits[32];
    size_t n = 0;
    unsigned long magnitude = value < 0 ? (unsigned long)(-value) : (unsigned long)value;
    do {
        digits[n++] = (char)('0' + magnitude % 10);
        magnitude /= 10;
    } while (magnitude != 0 && n < sizeof digits);
    if (value < 0) {
        put(line, "-");
    }
    while (n > 0) {
        char one[2] = {digits[--n], '\0'};
        put(line, one);
    }
}

static void put_names(struct line *line, char *const envp[], const char *prefix) {
    size_t prefix_len = strlen(prefix);
    int first = 1;
    for (size_t i = 0; envp != NULL && envp[i] != NULL; i++) {
        if (strncmp(envp[i], prefix, prefix_len) != 0) {
            continue;
        }
        const char *eq = strchr(envp[i], '=');
        size_t name_len = eq ? (size_t)(eq - envp[i]) : strlen(envp[i]);
        if (!first) {
            put(line, ",");
        }
        first = 0;
        char name[256];
        if (name_len >= sizeof name) {
            name_len = sizeof name - 1;
        }
        memcpy(name, envp[i], name_len);
        name[name_len] = '\0';
        put(line, name);
    }
    if (first) {
        put(line, "-");
    }
}

static void log_exec(const char *how, const char *path, char *const argv[], char *const envp[],
                     int routed) {
    const char *log_path = getenv("X86SHIM_LOG");
    if (log_path == NULL || log_path[0] == '\0') {
        return;
    }
    struct line line = {.len = 0};
    put(&line, "X86SHIM ");
    put(&line, how);
    put(&line, " pid=");
    put_number(&line, (long)getpid());
    put(&line, " routed=");
    put_number(&line, routed);
    put(&line, " path=");
    put(&line, path);
    put(&line, " arg1=");
    put(&line, (argv != NULL && argv[0] != NULL && argv[1] != NULL) ? argv[1] : "-");
    put(&line, " env_ny=");
    put_names(&line, envp, "NY_");
    put(&line, " env_ay=");
    put_names(&line, envp, "AY_");
    put(&line, " ld_preload=");
    int preload = 0;
    for (size_t i = 0; envp != NULL && envp[i] != NULL; i++) {
        if (strncmp(envp[i], "LD_PRELOAD=", 11) == 0) {
            preload = 1;
        }
    }
    put_number(&line, preload);
    line.text[line.len++] = '\n';
    int fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd >= 0) {
        ssize_t ignored = write(fd, line.text, line.len);
        (void)ignored;
        close(fd);
    }
}

// Build `qemu -cpu C [-L S] -0 argv0 path argv1..` into out. Returns 0 or -1.
static int qemu_argv(const char *path, char *const argv[], char **out, size_t cap) {
    const char *qemu = getenv("X86SHIM_QEMU");
    const char *cpu = getenv("X86SHIM_QEMU_CPU");
    const char *sysroot = getenv("X86SHIM_QEMU_SYSROOT");
    size_t n = 0;
    out[n++] = (char *)qemu;
    out[n++] = (char *)"-cpu";
    out[n++] = (char *)((cpu != NULL && cpu[0] != '\0') ? cpu : "Haswell-v4");
    if (sysroot != NULL && sysroot[0] != '\0') {
        out[n++] = (char *)"-L";
        out[n++] = (char *)sysroot;
    }
    out[n++] = (char *)"-0";
    out[n++] = (char *)((argv != NULL && argv[0] != NULL) ? argv[0] : path);
    out[n++] = (char *)path;
    for (size_t i = 1; argv != NULL && argv[0] != NULL && argv[i] != NULL; i++) {
        if (n + 1 >= cap) {
            errno = E2BIG;
            return -1;
        }
        out[n++] = argv[i];
    }
    out[n] = NULL;
    return 0;
}

static int shim_execve(const char *path, char *const argv[], char *const envp[]) {
    execve_fn next = real_execve();
    if (next == NULL) {
        errno = ENOSYS;
        return -1;
    }
    if (!routes(path)) {
        if (is_subject(path)) {
            char *clean[SHIM_MAX_ENV];
            guest_env(envp, clean, SHIM_MAX_ENV);
            log_exec("exec", path, argv, clean, 0);
            return next(path, argv, clean);
        }
        log_exec("exec", path, argv, envp, 0);
        return next(path, argv, envp);
    }
    char *guest[SHIM_MAX_ENV];
    char *qargv[SHIM_MAX_ARGS];
    guest_env(envp, guest, SHIM_MAX_ENV);
    if (qemu_argv(path, argv, qargv, SHIM_MAX_ARGS) != 0) {
        return -1;
    }
    log_exec("exec", path, argv, guest, 1);
    return next(qargv[0], qargv, guest);
}

int execve(const char *path, char *const argv[], char *const envp[]) {
    return shim_execve(path, argv, envp);
}

int execv(const char *path, char *const argv[]) {
    return shim_execve(path, argv, environ);
}

// glibc's execvp/execvpe call the INTERNAL __execve, which LD_PRELOAD cannot
// see, so the PATH search is re-implemented here with glibc's semantics:
// skip ENOENT/ENOTDIR/EACCES candidates (remembering EACCES), run a
// shebang-less script through /bin/sh on ENOEXEC.
int execvpe(const char *file, char *const argv[], char *const envp[]) {
    if (file[0] == '\0') {
        errno = ENOENT;
        return -1;
    }
    if (strchr(file, '/') != NULL) {
        int rc = shim_execve(file, argv, envp);
        if (errno == ENOEXEC) {
            char *sh_argv[SHIM_MAX_ARGS];
            size_t n = 0;
            sh_argv[n++] = (char *)"/bin/sh";
            sh_argv[n++] = (char *)file;
            for (size_t i = 1; argv[0] != NULL && argv[i] != NULL && n + 1 < SHIM_MAX_ARGS; i++) {
                sh_argv[n++] = argv[i];
            }
            sh_argv[n] = NULL;
            return shim_execve("/bin/sh", sh_argv, envp);
        }
        return rc;
    }
    const char *search = getenv("PATH");
    if (search == NULL) {
        search = "/bin:/usr/bin";
    }
    int saw_eacces = 0;
    size_t file_len = strlen(file);
    const char *cursor = search;
    for (;;) {
        const char *end = strchr(cursor, ':');
        size_t dir_len = end ? (size_t)(end - cursor) : strlen(cursor);
        char candidate[PATH_MAX];
        if (dir_len + 1 + file_len + 1 <= sizeof candidate) {
            if (dir_len == 0) {
                memcpy(candidate, file, file_len + 1);
            } else {
                memcpy(candidate, cursor, dir_len);
                candidate[dir_len] = '/';
                memcpy(candidate + dir_len + 1, file, file_len + 1);
            }
            shim_execve(candidate, argv, envp);
            if (errno == EACCES) {
                saw_eacces = 1;
            } else if (errno == ENOEXEC) {
                char *sh_argv[SHIM_MAX_ARGS];
                size_t n = 0;
                sh_argv[n++] = (char *)"/bin/sh";
                sh_argv[n++] = candidate;
                for (size_t i = 1; argv[0] != NULL && argv[i] != NULL && n + 1 < SHIM_MAX_ARGS;
                     i++) {
                    sh_argv[n++] = argv[i];
                }
                sh_argv[n] = NULL;
                return shim_execve("/bin/sh", sh_argv, envp);
            } else if (errno != ENOENT && errno != ENOTDIR && errno != ELOOP
                       && errno != ENAMETOOLONG) {
                return -1;
            }
        }
        if (end == NULL) {
            break;
        }
        cursor = end + 1;
    }
    errno = saw_eacces ? EACCES : ENOENT;
    return -1;
}

int execvp(const char *file, char *const argv[]) {
    return execvpe(file, argv, environ);
}

#define SHIM_COLLECT_ARGS(first, args, out)                                                    \
    do {                                                                                       \
        size_t n_ = 0;                                                                         \
        (out)[n_++] = (char *)(first);                                                         \
        char *next_;                                                                           \
        while ((next_ = va_arg(args, char *)) != NULL) {                                       \
            if (n_ + 1 >= SHIM_MAX_ARGS) {                                                     \
                errno = E2BIG;                                                                 \
                return -1;                                                                     \
            }                                                                                  \
            (out)[n_++] = next_;                                                               \
        }                                                                                      \
        (out)[n_] = NULL;                                                                      \
    } while (0)

int execl(const char *path, const char *arg, ...) {
    char *argv[SHIM_MAX_ARGS];
    va_list args;
    va_start(args, arg);
    SHIM_COLLECT_ARGS(arg, args, argv);
    va_end(args);
    return shim_execve(path, argv, environ);
}

int execlp(const char *file, const char *arg, ...) {
    char *argv[SHIM_MAX_ARGS];
    va_list args;
    va_start(args, arg);
    SHIM_COLLECT_ARGS(arg, args, argv);
    va_end(args);
    return execvpe(file, argv, environ);
}

int execle(const char *path, const char *arg, ...) {
    char *argv[SHIM_MAX_ARGS];
    va_list args;
    va_start(args, arg);
    SHIM_COLLECT_ARGS(arg, args, argv);
    char *const *envp = va_arg(args, char *const *);
    va_end(args);
    return shim_execve(path, argv, envp);
}

static int shim_spawn(const char *which, pid_t *pid, const char *path,
                      const posix_spawn_file_actions_t *actions, const posix_spawnattr_t *attrs,
                      char *const argv[], char *const envp[]) {
    spawn_fn next = real_spawn(which);
    if (next == NULL) {
        return ENOSYS;
    }
    // posix_spawnp: resolve a bare name the way the shell would before
    // deciding whether it names an x86_64 ELF.
    char resolved[PATH_MAX];
    const char *target = path;
    if (strcmp(which, "posix_spawnp") == 0 && path != NULL && strchr(path, '/') == NULL) {
        target = NULL;
        const char *search = getenv("PATH");
        const char *cursor = search ? search : "/bin:/usr/bin";
        size_t file_len = strlen(path);
        for (;;) {
            const char *end = strchr(cursor, ':');
            size_t dir_len = end ? (size_t)(end - cursor) : strlen(cursor);
            if (dir_len > 0 && dir_len + 1 + file_len + 1 <= sizeof resolved) {
                memcpy(resolved, cursor, dir_len);
                resolved[dir_len] = '/';
                memcpy(resolved + dir_len + 1, path, file_len + 1);
                if (access(resolved, X_OK) == 0) {
                    target = resolved;
                    break;
                }
            }
            if (end == NULL) {
                break;
            }
            cursor = end + 1;
        }
    }
    if (target == NULL || !routes(target)) {
        log_exec(which, target ? target : path, argv, envp, 0);
        return next(pid, path, actions, attrs, argv, envp);
    }
    char *guest[SHIM_MAX_ENV];
    char *qargv[SHIM_MAX_ARGS];
    guest_env(envp, guest, SHIM_MAX_ENV);
    if (qemu_argv(target, argv, qargv, SHIM_MAX_ARGS) != 0) {
        return E2BIG;
    }
    log_exec(which, target, argv, guest, 1);
    spawn_fn direct = real_spawn("posix_spawn");
    if (direct == NULL) {
        return ENOSYS;
    }
    return direct(pid, qargv[0], actions, attrs, qargv, guest);
}

int posix_spawn(pid_t *pid, const char *path, const posix_spawn_file_actions_t *actions,
                const posix_spawnattr_t *attrs, char *const argv[], char *const envp[]) {
    return shim_spawn("posix_spawn", pid, path, actions, attrs, argv, envp);
}

int posix_spawnp(pid_t *pid, const char *file, const posix_spawn_file_actions_t *actions,
                 const posix_spawnattr_t *attrs, char *const argv[], char *const envp[]) {
    return shim_spawn("posix_spawnp", pid, file, actions, attrs, argv, envp);
}

int uname(struct utsname *buf) {
    uname_fn next = (uname_fn)dlsym(RTLD_NEXT, "uname");
    if (next == NULL) {
        errno = ENOSYS;
        return -1;
    }
    int rc = next(buf);
    const char *machine = getenv("X86SHIM_UNAME_MACHINE");
    if (rc == 0 && machine != NULL && machine[0] != '\0') {
        size_t len = strlen(machine);
        if (len >= sizeof buf->machine) {
            len = sizeof buf->machine - 1;
        }
        memcpy(buf->machine, machine, len);
        buf->machine[len] = '\0';
    }
    return rc;
}
