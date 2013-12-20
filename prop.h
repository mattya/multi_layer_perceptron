#ifndef __prop
#define __prop


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
#include <cublas.h>

#include "kernel_misc.h"

using namespace std;

extern float beta, lambda, eta0;
extern int pitch_x, pitch_y;

extern int batch_size;
extern int NTrain;
extern int NTest;
extern int N_layer;     // include input and output layer
extern int *N_neuron;
extern int N, M, N_max;        // input and output layer
extern float **x_cpu;
extern float **w_cpu;
extern float **data_test, **data_train;
extern float **label_test, **label_train;

extern float **gpu_data_test, **gpu_data_train, **gpu_label_test, **gpu_label_train;
extern float **x_gpu, **w_gpu, **delta_gpu;

extern float *curnd;
extern curandGenerator_t curand_gen;

__global__ set_one(float *x, int n){
	x[n] = 1.0f;
}

void forward_prop(float *l1, float *w, float *l2, int n1, int n2){
	set_one<<<1, 1>>>(l1, n1-1);
//	set_zero<<<n2/pitch_x, pitch_x>>>(l2);
	cublasSgemv('n', n2, n1, 1.0, w, l1, 1, 0.0, l2, 1);
	sigmoid<<<n2, 1>>>(l2, l2);
}

void back_prop(float *l1, float *d1, float *w, float *d2, int n1, int n2){
	cublasSgemv('t', n2, n1-1, 1.0, w, d2, 1, 0.0, d1, 1);
	dsigmoid<<<n1, 1>>>(l1, d1);
}

void update_weights(float *l1, float *w, float *d2, float eta, int n1, int n2){
	set_one<<<1, 1>>>(l1, n1-1);
	cublasSger(n2, n1, -eta, d2, 1, l1, 1, w, n2);
}


#endif /* __prop */