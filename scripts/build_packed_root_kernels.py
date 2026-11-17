#!/usr/bin/env python3
"""Offline native CUDA images and literal-bit KAT pack for packed root arithmetic.

Usage: build-slot -- python3 scripts/build_packed_root_kernels.py /path/libnvrtc.so
No NVRTC or PTX is used at solver runtime. The committed oracle evidence is the
source of fixture expected bits; this builder never derives them from the GPU.
"""
from pathlib import Path
import ctypes as C
import gzip
import hashlib
import json
import os
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'crates/ny-cuda/kernels'
SOURCE = OUT / 'packed_root_conv.cu'
FLAGS = ['--std=c++17', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true']
ARCHES = (86, 120, 121)
EVIDENCE = ROOT / 'reports/measured-2026/packed-root-cuda-prototype-20261117/experiment/device-probe-sm120-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture_pack():
    raw = gzip.decompress((EVIDENCE / 'fixtures.json.gz').read_bytes())
    expected_raw = gzip.decompress((EVIDENCE / 'expected.json.gz').read_bytes())
    assert sha(raw) == '2031edfc908b859734db41df164b23eee1d4f1c7a899a04ea297608c0315c208'
    assert sha(expected_raw) == '92a636f0bd5642012e67492a843a36ef9af02e4cf98d24e6abe8da2f704cb840'
    fixtures, expected = json.loads(raw), json.loads(expected_raw)
    assert len(fixtures) == len(expected) == 9
    output = bytearray(b'NYPRKAT1' + struct.pack('<I', len(fixtures)))
    def vector(values):
        output.extend(struct.pack('<I', len(values)))
        output.extend(struct.pack('<' + 'I' * len(values), *values))
    for f in fixtures:
        output.extend(struct.pack('<6I', f['source_stride'], f['output_stride'],
                      f['oh'] * f['ow'], f['co'], f['kh'] * f['kw'], f['ci'] * f['kh'] * f['kw']))
        offsets, runs = [0], []
        for bucket in f['runs']:
            runs.extend(bucket)
            offsets.append(len(runs))
        for values in (f['source_m'], f['source_d'], f['weights'], f['gather'],
                       offsets, [x for run in runs for x in run], *expected[f['name']]):
            vector(values)
    return bytes(output)


def main():
    if not __debug__:
        raise RuntimeError('builder requires enabled assertions')
    library = Path(sys.argv[1]).resolve(strict=True)
    source = SOURCE.read_bytes()
    fixtures = fixture_pack()
    compiler_files = {str(p.resolve()): sha(p.read_bytes())
                      for p in library.parent.glob('libnvrtc*.so*') if p.is_file()}
    assert str(library) in compiler_files
    nv = C.CDLL(str(library))
    def bind(name, args):
        fn = getattr(nv, name); fn.argtypes = args; fn.restype = C.c_int
        return fn
    version = bind('nvrtcVersion', [C.POINTER(C.c_int), C.POINTER(C.c_int)])
    create = bind('nvrtcCreateProgram', [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                        C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)])
    compile_fn = bind('nvrtcCompileProgram', [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)])
    size_log = bind('nvrtcGetProgramLogSize', [C.c_void_p, C.POINTER(C.c_size_t)])
    get_log = bind('nvrtcGetProgramLog', [C.c_void_p, C.c_void_p])
    size_bin = bind('nvrtcGetCUBINSize', [C.c_void_p, C.POINTER(C.c_size_t)])
    get_bin = bind('nvrtcGetCUBIN', [C.c_void_p, C.c_void_p])
    destroy = bind('nvrtcDestroyProgram', [C.POINTER(C.c_void_p)])
    def check(code):
        if code: raise RuntimeError(f'NVRTC returned {code}')
    major, minor = C.c_int(), C.c_int()
    check(version(C.byref(major), C.byref(minor)))
    images, logs = {}, {}
    for arch in ARCHES:
        program = C.c_void_p()
        check(create(C.byref(program), source, b'packed_root_conv.cu', 0, None, None))
        try:
            flags = FLAGS + [f'--gpu-architecture=sm_{arch}']
            args = (C.c_char_p * len(flags))(*(s.encode() for s in flags))
            code = compile_fn(program, len(flags), args)
            size = C.c_size_t(); check(size_log(program, C.byref(size)))
            assert 0 < size.value < 16 * 1024**2
            log = C.create_string_buffer(size.value); check(get_log(program, log))
            logs[str(arch)] = log.value.decode()
            if code: print(logs[str(arch)], file=sys.stderr)
            check(code)
            check(size_bin(program, C.byref(size)))
            assert 0 < size.value < 64 * 1024**2
            image = C.create_string_buffer(size.value); check(get_bin(program, image))
            assert image.raw.startswith(b'\x7fELF')
            images[arch] = image.raw
        finally:
            check(destroy(C.byref(program)))
    assert source == SOURCE.read_bytes() and fixtures == fixture_pack()
    assert all(sha(Path(p).read_bytes()) == h for p, h in compiler_files.items())
    loaded = {str(Path(line.split()[-1]).resolve()) for line in Path('/proc/self/maps').read_text().splitlines()
              if 'libnvrtc' in line and line.split()[-1].startswith('/')}
    assert str(library) in loaded and loaded.issubset(compiler_files)
    manifest = ['format=1', 'entry=packed_root_conv_ordered', 'abi=seven-pointers-six-u32-two-u64',
                'source_sha256=' + sha(source), 'fixtures_sha256=' + sha(fixtures),
                f'compiler=NVRTC {major.value}.{minor.value}', 'flags=' + ' '.join(FLAGS),
                'runtime_compilation=false', 'ptx_fallback=false']
    manifest.extend(f'sm_{arch}_sha256={sha(image)}' for arch, image in images.items())
    record = {'compiler_files': compiler_files, 'loaded_compiler_files': sorted(loaded),
              'source_sha256': sha(source), 'fixture_sha256': sha(fixtures),
              'compiler_logs': logs, 'flags': FLAGS, 'architectures': list(ARCHES),
              'builder_sha256': sha(Path(__file__).read_bytes()), 'runtime_qualification_claimed': False}
    with tempfile.TemporaryDirectory(dir=OUT, prefix='.packed-root-build-') as temp:
        temp = Path(temp)
        for arch, image in images.items():
            (temp / f'packed_root_conv.sm{arch}.cubin').write_bytes(image)
        (temp / 'packed_root_kat.bin').write_bytes(fixtures)
        (temp / 'packed_root_kernels.manifest').write_text('\n'.join(manifest) + '\n')
        (temp / 'packed_root_kernels.build.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        for path in sorted(temp.iterdir()): os.replace(path, OUT / path.name)
    print(json.dumps({'compiled_architectures': list(images), 'source_sha256': sha(source),
                      'fixture_bytes': len(fixtures), 'runtime_qualification_claimed': False}))


if __name__ == '__main__':
    main()
