#ifndef __train
#define __train


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
#include "prop.h"

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

void random_init(){
	for(int i=0; i<N_layer-1; i++){
		for(int j=0; j<N_neuron[i]*N_neuron[i+1]; j++){
			w_cpu[i][j] = random(-0.1, 0.1);
		}
	}
	cerr << "random_init done" << endl;
}


void train_step(int ind, int lp){
	set_state<<<N, 1>>>(x_gpu[0], gpu_data_train[ind]);
	for(int i=0; i<N_layer-1; i++){
		forward_prop(x_gpu[i], w_gpu[i], x_gpu[i+1], N_neuron[i]+1, N_neuron[i+1]);
	}
	set_delta<<<M, 1>>>(gpu_label_train[ind], x_gpu[N_layer-1], delta_gpu[N_layer-1]);
	for(int i=N_layer-2; i>=1; i--){
		back_prop(x_gpu[i], delta_gpu[i], w_gpu[i], delta_gpu[i+1], N_neuron[i]+1, N_neuron[i+1]);
	}
	for(int i=0; i<N_layer-1; i++){
		update_weights(x_gpu[i],  w_gpu[i], delta_gpu[i+1], eta0, N_neuron[i]+1, N_neuron[i+1]);
	}
//	cerr << "train_step done" << endl;
/*
	for(int i=0; i<N_layer; i++){
		float *cpu_x = (float *)malloc(N_neuron[i]*sizeof(float));
		cudaMemcpy(cpu_x, x_gpu[i], N_neuron[i]*sizeof(float), cudaMemcpyDeviceToHost);
		for(int j=0; j<N_neuron[i]; j++){
			cerr << cpu_x[j] << " ";
		}
		cerr << endl;
	}
	cerr << endl;
	for(int i=N_layer-1; i>0; i--){
		float *cpu_x = (float *)malloc(N_neuron[i]*sizeof(float));
		cudaMemcpy(cpu_x, delta_gpu[i], N_neuron[i]*sizeof(float), cudaMemcpyDeviceToHost);
		for(int j=0; j<N_neuron[i]; j++){
			cerr << cpu_x[j] << " ";
		}
		cerr << endl;
	}
	cerr << endl;

	cerr << "--------" << endl;
	*/
}

void train_error(){
	float err = 0;
	// for each data
	for(int d=0; d<batch_size; d++){
		set_state<<<N, 1>>>(x_gpu[0], gpu_data_train[d]);
		for(int i=0; i<N_layer-1; i++){
			forward_prop(x_gpu[i], w_gpu[i], x_gpu[i+1], N_neuron[i]+1, N_neuron[i+1]);
		}
		set_delta<<<M, 1>>>(gpu_label_train[d], x_gpu[N_layer-1], delta_gpu[N_layer-1]);

		float *cpu_delta = (float *)malloc(M*sizeof(float));
		cudaMemcpy(cpu_delta, delta_gpu[N_layer-1], M*sizeof(float), cudaMemcpyDeviceToHost);
		
		float sum = 0;
		for(int k=0; k<M; k++){
			if(cpu_delta[k]*cpu_delta[k] > 0.25) sum = 1.0;
//			sum += cpu_delta[k]*cpu_delta[k]/M;
		}
		err += sum;

		free(cpu_delta);
	}
	printf("train err: %f\n", err/batch_size);
	cerr << "train err: " << err/batch_size << endl;
}

void test_error(){
	float err = 0;
	// for each data
	for(int d=0; d<NTest; d++){
		set_state<<<N, 1>>>(x_gpu[0], gpu_data_test[d]);
		for(int i=0; i<N_layer-1; i++){
			forward_prop(x_gpu[i], w_gpu[i], x_gpu[i+1], N_neuron[i]+1, N_neuron[i+1]);
		}
		set_delta<<<M, 1>>>(gpu_label_test[d], x_gpu[N_layer-1], delta_gpu[N_layer-1]);

		float *cpu_delta = (float *)malloc(M*sizeof(float));
		cudaMemcpy(cpu_delta, delta_gpu[N_layer-1], M*sizeof(float), cudaMemcpyDeviceToHost);
		
		float sum = 0;
		for(int k=0; k<M; k++){
			if(cpu_delta[k]*cpu_delta[k] > 0.25) sum = 1.0;
//			sum += cpu_delta[k]*cpu_delta[k]/M;
		}
		err += sum;

		free(cpu_delta);
	}
	printf("test err: %f\n", err/NTest);
	cerr << "test err: " << err/NTest << endl;
}


#endif /* __train */