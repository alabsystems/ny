#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
static void one(const char*name,double a,double b,double expect){
  double A[1]={a},B[1]={b},C[1]={0};
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,1,1.0,A,1,B,1,0.0,C,1);
  int ok = (C[0]==expect) || (isnan(C[0])&&isnan(expect));
  printf("  %-42s got=%-24.17g want=%-24.17g %s\n",name,C[0],expect,ok?"OK":"MISMATCH");
}
int main(void){
  double dmin=0x1p-1022;            /* smallest normal   */
  double sub =0x1p-1074;            /* smallest subnormal*/
  double maxsub=0x0.fffffffffffffp-1022;
  puts("Accelerate cblas_dgemm conformance (f64):");
  one("DAZ? subnormal operand * 1.0", sub, 1.0, sub);
  one("DAZ? largest subnormal * 1.0", maxsub, 1.0, maxsub);
  one("FTZ? dmin * 2^-1 = subnormal", dmin, 0.5, dmin*0.5);
  one("FTZ? normal*normal -> subnormal", 0x1p-600, 0x1p-500, 0x1p-1100);
  one("exact integer product", 3.0, 7.0, 21.0);
  one("finite sentinel 1e30 * 1.0", 1e30, 1.0, 1e30);
  one("large finite 1e300 * 1.0", 1e300, 1.0, 1e300);
  /* accumulation exactness: sum of 1/2^i pairs, k=64, all exactly representable */
  {
    int k=64; double A[64],B[64],C[1]={0}; double ref=0;
    for(int i=0;i<k;i++){A[i]=ldexp(1.0,-(i%40)); B[i]=ldexp(1.0,-(i%40)); ref+=A[i]*B[i];}
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,k,1.0,A,k,B,1,0.0,C,1);
    printf("  %-42s got=%-24.17g ref(serial)=%-24.17g rel=%.3e\n","k=64 dyadic dot",C[0],ref,fabs(C[0]-ref)/fabs(ref));
  }
  return 0;
}
