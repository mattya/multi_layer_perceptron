
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
#include "file_manager.h"
#include "memory_alloc.h"
#include "train.h"

using namespace std;

float beta = 10.0;
float lambda = 0.00000001;
float eta0 = 0.015;
int pitch_x = 32, pitch_y = 32;

int batch_size = 1000;
int NTrain = 10000;
int NTest = 10000;

int N_layer;     // include input and output layer
int *N_neuron;
int N, M, N_max;        // input and output layer
float **x_cpu;
float **w_cpu;
float **data_test, **data_train;
float **label_test, **label_train;

float **gpu_data_test, **gpu_data_train, **gpu_label_test, **gpu_label_train;
float **x_gpu, **w_gpu, **delta_gpu;

float *curnd;
curandGenerator_t curand_gen;


void learning(){

	// transfer data
	random_init();
	cpu_to_gpu_matrix();
	cpu_to_gpu_data_test();

	// add noise

	for(int loop=0; loop<5000; loop++){
		cerr << "loop: " << loop << endl;
		for(int ib=0; ib<NTrain/batch_size; ib++){
			printf("loop, batch: %d, %d\n", loop, ib);
			cpu_to_gpu_data_train(ib*batch_size);
//			deform_image();

			// for each data
			for(int i=0; i<batch_size; i++){
			//	train_step
				train_step(i, loop);
			}

			// calc train error
			train_error();
		}

		// calc test error
		test_error();

		output_weight();
	}
}
int main(){
	cublasInit();

	load_data();
	gpu_alloc();
	learning();

	gpu_free();
	return 0;
}