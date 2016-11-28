// Code adapted from MATLAB implementation at https://people.ece.cornell.edu/land/courses/ece5760/LABS/s2016/lab3.html
#include <stdio.h>
#include <stdlib.h>
#define N 512 // grid side length
#define RHO 0.5 // related to pitch
#define ETA 2e-4 // related to duration of sound
#define BOUNDARY_GAIN 0.75 // clamped edge vs free edge

__global__ void iterate_grid(float* d_u0, float* d_u1, float* d_u2, float* d_audio, int iteration) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float sum_of_neighbors, previous_value, previous_previous_value;

	//inner points
	if(idx%N!=0 && idx%N!=(N-1) && idx > N-1 && idx < N*N - N ) {
		sum_of_neighbors = d_u1[idx+N] + d_u1[idx-N] + d_u1[idx-1] + d_u1[idx+1];
		previous_value = d_u1[idx];
		previous_previous_value = d_u2[idx];
		d_u0[idx] = (RHO * (sum_of_neighbors-4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
	}

	//side points

	if(idx%N==0 && idx!=0 && idx!=N*N-N) {
		int neighbor_idx = idx + 1;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		d_u0[idx] = BOUNDARY_GAIN * temp;
	}

	if(idx%N==N-1 && idx!= N-1 && idx!= N*N-1 ) {
		int neighbor_idx = idx - 1;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		d_u0[idx] = BOUNDARY_GAIN * temp;
	}

	if(idx<N-1 && idx>0) {
		int neighbor_idx = idx + N;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		d_u0[idx] = BOUNDARY_GAIN * temp;
	}

	if(idx<N*N-1 && idx>N*N-N) {
		int neighbor_idx = idx - N;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		d_u0[idx] = BOUNDARY_GAIN * temp;
	}

	if(idx==0) {
		//look down and get neight
		int neighbor_idx = idx + N + 1;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		float temp1 = BOUNDARY_GAIN * temp;
		d_u0[idx] = BOUNDARY_GAIN * temp1;
	}

	if(idx==N-1) {
		//look left and get neighbor
		int neighbor_idx = idx - 1 + N;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		float temp1 = BOUNDARY_GAIN * temp;
		d_u0[idx] = BOUNDARY_GAIN * temp1;
	}
	
	if(idx==N*N-1) {
		//look left and get neighbor
		int neighbor_idx = idx - 1 - N;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		float temp1 = BOUNDARY_GAIN * temp;
		d_u0[idx] = BOUNDARY_GAIN * temp1;
	}

	if(idx==N*N-N) {
		//look up and get neighbor
		int neighbor_idx = idx - N + 1;
		sum_of_neighbors = d_u1[neighbor_idx+N] + d_u1[neighbor_idx-N] + d_u1[neighbor_idx-1] + d_u1[neighbor_idx+1];
		previous_value = d_u1[neighbor_idx];
		previous_previous_value = d_u2[neighbor_idx];
		float temp = (RHO * (sum_of_neighbors -4*previous_value) + 2*previous_value -(1-ETA)*previous_previous_value)/(1+ETA);
		float temp1 = BOUNDARY_GAIN * temp;
		d_u0[idx] = BOUNDARY_GAIN * temp1;
	}

	if(idx==(N*N/2+N/2)) {
		d_audio[iteration] = d_u1[idx];
	}

}

void print_grid(float **grid) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("(%d,%d): %f ", i,j,grid[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char** argv) {
	// get number of iterations to perform
	int T = atoi(argv[1]);

	// initialize grid
	const int GRID_BYTES = N*N*sizeof(float);
	const int AUDIO_BYTES = T * sizeof(float);
	float* h_u0 = (float*) malloc(GRID_BYTES);
	float* h_u1 = (float*) malloc(GRID_BYTES);
	float* h_u2 = (float*) malloc(GRID_BYTES);
	float* h_audio = (float*) malloc(AUDIO_BYTES);


	printf("Size of grid: %d nodes\n", N*N);

	for(int i = 0; i<N*N; i++) {
		h_u0[i] = 0.0;
		h_u1[i] = 0.0;
		h_u2[i] = 0.0;
	}

	// for(int i = 0; i<T;i++) {
	// 	h_audio[i] = 0.0;

	// }

	h_u1[N*N/2 + N/2] = 1.0;

	float* d_u0;
	float* d_u1;
	float* d_u2;
	float* d_audio;

    cudaMalloc(&d_u0, GRID_BYTES);
    cudaMalloc(&d_u1, GRID_BYTES);
    cudaMalloc(&d_u2, GRID_BYTES);
    cudaMalloc(&d_audio, AUDIO_BYTES);

    cudaMemcpy(d_u0, h_u0, GRID_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u1, h_u1, GRID_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2, h_u2, GRID_BYTES, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_audio, h_audio, AUDIO_BYTES, cudaMemcpyHostToDevice);



    //launch kernel(s)
    int block_size = 1024;
    printf("block size %d \n", 1024);
    for(int i=0; i<T ; i++) {
    	iterate_grid<<<block_size, N*N/block_size>>> (d_u0, d_u1, d_u2, d_audio, i);
    	float* temp = d_u2;
    	d_u2 = d_u1;
    	d_u1 = d_u0;
  		d_u0 = temp;
	}

    cudaMemcpy(h_audio, d_audio, AUDIO_BYTES, cudaMemcpyDeviceToHost);

    for(int i=0; i< T;i++) {
    	printf("%f \n", h_audio[i]);

    }

	free(h_u0);
	free(h_u1);
	free(h_u2);
	free(h_audio);
}
