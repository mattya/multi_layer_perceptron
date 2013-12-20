#ifndef __file_manager
#define __file_manager

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

#include "memory_alloc.h"


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

void load_data(){
	FILE *fp = fopen("architecture.txt", "r");
	fscanf(fp, "%d", &N_layer);
	N_neuron = (int *)malloc(N_layer*sizeof(int));
	for(int i=0; i<N_layer; i++){
		int tmp_n;
		fscanf(fp, "%d", &tmp_n);
		N_neuron[i] = tmp_n;
	}
	fclose(fp);

	N = N_neuron[0];
	M = N_neuron[N_layer-1];
	N_max = 2000;

	data_train = (float **)malloc(NTrain*sizeof(float *));
	label_train = (float **)malloc(NTrain*sizeof(float *));
	for(int i=0; i<NTrain; i++){
		data_train[i] = (float *)malloc(N*sizeof(float));
		label_train[i] = (float *)malloc(M*sizeof(float));
	}
	data_test = (float **)malloc(NTest*sizeof(float *));
	label_test = (float **)malloc(NTest*sizeof(float *));
	for(int i=0; i<NTest; i++){
		data_test[i] = (float *)malloc(N*sizeof(float));
		label_test[i] = (float *)malloc(M*sizeof(float));
	}

	x_cpu = (float **)malloc(N_layer*sizeof(float *));
	w_cpu = (float **)malloc(N_layer*sizeof(float *));
	for(int i=0; i<N_layer; i++){
		x_cpu[i] = (float *)malloc(N_neuron[i]*sizeof(float));
	}
	for(int i=0; i<N_layer-1; i++){
		w_cpu[i] = (float *)malloc((N_neuron[i]+1)*N_neuron[i+1]*sizeof(float));
	}

	fp = fopen("../make_training_data/my_output.txt", "r");
	for(int i=0; i<NTrain; i++){
		for(int j=0; j<N; j++){
			int in;
			fscanf(fp, "%d", &in);
			data_train[i][j] = (float)in;
		}
	}
	for(int i=0; i<NTest; i++){
		for(int j=0; j<N; j++){
			int in;
			fscanf(fp, "%d", &in);
			data_test[i][j] = (float)in;
		}
	}
	fclose(fp);

	fp = fopen("../make_training_data/my_input.txt", "r");
	for(int i=0; i<NTrain; i++){
		int tmp;
		for(int j=0; j<M; j++){
			fscanf(fp, "%d", &tmp);
			label_train[i][j] = (float)tmp;
		}
	}
	for(int i=0; i<NTest; i++){
		int tmp;
		for(int j=0; j<M; j++){
			fscanf(fp, "%d", &tmp);
			label_test[i][j] = (float)tmp;
		}
	}
	fclose(fp);

	cerr << "loading done" << endl;
}

void output_weight(){
	gpu_to_cpu_matrix();
	for(int i=0; i<N_layer-1; i++){
		char fn[100];
		sprintf(fn, "out_weight_%d.txt", i);
		FILE *fp = fopen(fn, "w");
		for(int j=0; j<N_neuron[i+1]; j++){
			for(int k=0; k<N_neuron[i]+1; k++){
				fprintf(fp, "%f ", w_cpu[i][k*N_neuron[i+1]+j]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	cerr << "output_weight done" << endl;
}


#endif /* __file_manager */