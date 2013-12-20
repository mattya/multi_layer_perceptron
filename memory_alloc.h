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

using namespace std;

extern float beta, lambda, eta0;
extern int pitch_x, pitch_y;

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


void gpu_alloc(){
	gpu_data_train = (float **)malloc(batch_size*sizeof(float *));
	gpu_label_train = (float **)malloc(batch_size*sizeof(float *));
	gpu_data_test = (float **)malloc(NTest*sizeof(float *));
	gpu_label_test = (float **)malloc(NTest*sizeof(float *));
	x_gpu = (float **)malloc(N_layer*sizeof(float *));
	delta_gpu = (float **)malloc(N_layer*sizeof(float *));
	w_gpu = (float **)malloc((N_layer-1)*sizeof(float *));

	for(int i=0; i<batch_size; i++){
		cudaMalloc((void**)&gpu_data_train[i], get_stride(N, pitch_x)*sizeof(float));
		cudaMalloc((void**)&gpu_label_train[i], get_stride(M, pitch_x)*sizeof(float));
	}
	for(int i=0; i<NTest; i++){
		cudaMalloc((void**)&gpu_data_test[i], get_stride(N, pitch_x)*sizeof(float));
		cudaMalloc((void**)&gpu_label_test[i], get_stride(M, pitch_x)*sizeof(float));
	}
	for(int i=0; i<N_layer; i++){
		cudaMalloc((void**)&x_gpu[i], (get_stride(N_neuron[i]+1, pitch_x))*sizeof(float));
		cudaMalloc((void**)&delta_gpu[i], (get_stride(N_neuron[i]+1, pitch_x))*sizeof(float));
	}
	for(int i=0; i<N_layer-1; i++){
		cudaMalloc((void**)&w_gpu[i], (get_stride(N_neuron[i]+1, pitch_y)*get_stride(N_neuron[i+1], pitch_x))*sizeof(float));
	}

	cudaMalloc((void**)&curnd, N*sizeof(float));
	curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curand_gen, 0);
	cerr << "gpu_alloc done" << endl;
}
void cpu_to_gpu_matrix(){
	for(int i=0; i<N_layer-1; i++){
//		set_zero<<<(get_stride(N_neuron[i]+1, pitch_y)*get_stride(N_neuron[i+1], pitch_x))/pitch_x, pitch_x>>>(w_gpu[i]);

		cublasSetMatrix(N_neuron[i+1], N_neuron[i]+1, sizeof(float), w_cpu[i], N_neuron[i+1], w_gpu[i], N_neuron[i+1]);

//		for(int j=0; j<N_neuron[i]+1; j++){
//			cudaMemcpy(&w_gpu[i][j*(get_stride(N_neuron[i+1], pitch_x))], &w_cpu[i][j*N_neuron[i+1]], N_neuron[i+1]*sizeof(float), cudaMemcpyHostToDevice);
//		}
	}
	cerr << "cpu_to_gpu_matrix done" << endl;
}
void cpu_to_gpu_data_train(int i0){
	for(int i=0; i<batch_size; i++){
		set_zero<<<get_stride(N, pitch_x)/pitch_x, pitch_x>>>(gpu_data_train[i]);
		set_zero<<<get_stride(M, pitch_x)/pitch_x, pitch_x>>>(gpu_label_train[i]);
	}
	for(int i=0; i<batch_size; i++){
		cudaMemcpy(&gpu_data_train[i][0], &data_train[i+i0][0], N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&gpu_label_train[i][0], &label_train[i+i0][0], M*sizeof(float), cudaMemcpyHostToDevice);
	}
	cerr << "cpu_to_gpu_data_train done" << endl;
}
void cpu_to_gpu_data_test(){
	for(int i=0; i<NTest; i++){
		set_zero<<<get_stride(N, pitch_x)/pitch_x, pitch_x>>>(gpu_data_test[i]);
		set_zero<<<get_stride(M, pitch_x)/pitch_x, pitch_x>>>(gpu_label_test[i]);
	}
	for(int i=0; i<NTest; i++){
		cudaMemcpy(&gpu_data_test[i][0], &data_test[i][0], N*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(&gpu_label_test[i][0], &label_test[i][0], M*sizeof(float), cudaMemcpyHostToDevice);
	}
	cerr << "cpu_to_gpu_data_test done" << endl;
}
void gpu_to_cpu_matrix(){
	for(int i=0; i<N_layer-1; i++){

		cublasGetmatrix(N_neuron[i+1], N_neuron[i]+1, sizeof(float), w_gpu[i], N_neuron[i+1], w_cpu[i], N_neuron[i+1]);

//		for(int j=0; j<N_neuron[i]+1; j++){
//			cudaMemcpy(&w_cpu[i][j*N_neuron[i+1]], &w_gpu[i][j*(get_stride(N_neuron[i+1], pitch_x))], N_neuron[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
//		}
	}
	cerr << "gpu_to_cpu_matrix done" << endl;
}