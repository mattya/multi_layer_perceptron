#ifndef __misc
#define __misc

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
using namespace std;

extern float beta, lambda, eta0;
extern int pitch_x, pitch_y;

extern int batch_size;
extern int NTrain;
extern int NTest;

__device__ float sigmoidk(float x){
	return 1.0/(1.0+exp(-beta*x));
}

__device__ float dsigmoidk(float x){
	return beta*x*(1.0-x);
}

__global__ void sigmoid(float *x, float *z){
	int idx = blockIdx.x*pitch_x + threadIdx.x;
	z[idx] = sigmoidk(x[idx]);
}

__global__ void sdigmoid(float *x, float *z){
	int idx = blockIdx.x*pitch_x + threadIdx.x;
	z[idx] *= dsigmoidk(x[idx]);
}

__device__ __host__ int get_stride(int n, int m){
	return n%m==0?n:m*(n/m)+m;
}

__global__ void set_state(float *z, float *x){
	int idx = blockIdx.x*pitch_x + threadIdx.x;
	x[idx] = z[idx];
}
__global__ void set_delta(float *z, float *x, float *delta){
	int idx = blockIdx.x*pitch_x + threadIdx.x;
	delta[idx] = x[idx] - z[idx];
}

__global__ void set_zero(float *x){
	x[blockIdx.x*pitch_x + threadIdx.x] = 0.0f;
}


#endif /* __misc */