#ifndef __noise
#define __noise

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

#define PI 3.141592653589

__global__ void affine_transform(float *img0, float s1, float s2, float r, float sk1, float sk2, float tx, float ty, float *cr){
	int id = threadIdx.x;
	int x0 = threadIdx.x%28;
	int y0 = threadIdx.x/28;
	__shared__ float img[28*28];

	img[id] = img0[id];
	__syncthreads();

	float A00 = s1*(cos(r));
	float A01 = s1*(-sin(r)+tan(sk1));
	float A10 = s2*(sin(r)+tan(sk2));
	float A11 = s2*(cos(r));

	int x1 = (int)(A00*(x0-14)+A01*(y0-14) + tx) + 14;
	int y1 = (int)(A10*(x0-14)+A11*(y0-14) + ty) + 14;

	if(x1>=0&&x1<28&&y1>=0&&y1<28){
		img0[id] = img[y1*28+x1];
	}else{
		img0[id] = 0;
	}

	if(cr[id]<0.09){
		img0[id] = 0;
	}else if(cr[id]<0.18){
		img0[id] = 1;
	}
}

void deform_image(){
	for(int i=0; i<batch_size; i++){
		curandGenerateUniform(curand_gen, curnd, 28*28);
		affine_transform<<<1, 28*28>>>(gpu_data_train[i], random(0.7, 1.2), random(0.7, 1.2), 
			random(-PI, PI), 
			random(-PI/6, PI/6), random(-PI/6, PI/6), 
			random(-5, 5), random(-5, 5),
			curnd);

		float rnd = random(0, 1);
		if(rnd<0.2){
//			enlarge<<<1, 28*28>>>(gpu_data_train[i], random(1, 3));
		}else if(rnd<0.4){
//			shrink<<<1, 28*28>>>(gpu_data_train[i], random(3, 6));
		}
	}
	cerr << "deform_image done" << endl;
}

#endif /* __noise */