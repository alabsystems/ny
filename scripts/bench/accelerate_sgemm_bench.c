#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(void){
  int shapes[4][3]={{512,64,512},{1024,128,1024},{2048,256,2048},{4096,512,4096}};
  printf("%-20s %10s %12s\n","shape m,k,n","ms","GFLOP/s f32");
  for(int s=0;s<4;s++){
    int m=shapes[s][0],k=shapes[s][1],n=shapes[s][2];
    float *A=malloc(sizeof(float)*(size_t)m*k);
    float *B=malloc(sizeof(float)*(size_t)k*n);
    float *C=malloc(sizeof(float)*(size_t)m*n);
    for(size_t i=0;i<(size_t)m*k;i++)A[i]=(float)rand()/RAND_MAX;
    for(size_t i=0;i<(size_t)k*n;i++)B[i]=(float)rand()/RAND_MAX;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,k,B,n,0.0f,C,n);
    int reps=5; double t0=now();
    for(int r=0;r<reps;r++) cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,k,B,n,0.0f,C,n);
    double dt=(now()-t0)/reps;
    char buf[32]; snprintf(buf,sizeof buf,"%d,%d,%d",m,k,n);
    printf("%-20s %10.2f %12.1f\n",buf,dt*1e3,2.0*(double)m*k*n/dt/1e9);
    free(A);free(B);free(C);
  }
  return 0;
}
