#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include "lodepng.h"

cudaError_t processWithCuda(unsigned char * h_input_image, unsigned char * h_output_image, const int IMAGE_BYTES, const int IMAGE_SIZE);

__global__ void rectify(unsigned char* d_output_image, unsigned char* d_input_image) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	d_output_image[4 * idx + 0] = d_input_image[4 * idx + 0] < 127 ? 127 : d_input_image[4 * idx + 0];
	d_output_image[4 * idx + 1] = d_input_image[4 * idx + 1] < 127 ? 127 : d_input_image[4 * idx + 1];
	d_output_image[4 * idx + 2] = d_input_image[4 * idx + 2] < 127 ? 127 : d_input_image[4 * idx + 2];
	d_output_image[4 * idx + 3] = 255;

}

void process(char *input_filename, char *output_filename, int num_threads) {
	unsigned error;
	unsigned char *h_input_image, *h_output_image;
	unsigned width, height;

	error = lodepng_decode32_file(&h_input_image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

	const int IMAGE_SIZE = width*height;
	const int IMAGE_BYTES = IMAGE_SIZE * 4 * sizeof(unsigned char);

	h_output_image = (unsigned char *)malloc(IMAGE_BYTES);

	cudaError_t cudaStatus = processWithCuda(h_input_image, h_output_image, IMAGE_BYTES, IMAGE_SIZE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed!");
	}

	clock_t begin = clock();
	
	int remainder = IMAGE_SIZE%1024;
	for (int idx = IMAGE_SIZE - remainder; idx < IMAGE_SIZE; idx++) {
		h_output_image[4 * idx + 0] = h_input_image[4 * idx + 0] < 127 ? 127 : h_input_image[4 * idx + 0];
		h_output_image[4 * idx + 1] = h_input_image[4 * idx + 1] < 127 ? 127 : h_input_image[4 * idx + 1];
		h_output_image[4 * idx + 2] = h_input_image[4 * idx + 2] < 127 ? 127 : h_input_image[4 * idx + 2];
		h_output_image[4 * idx + 3] = 255;
	}

	clock_t end = clock();
	double total = (double)(end - begin) / CLOCKS_PER_SEC * 1000.0;

	printf("runtime is %f ms \n", total);
	lodepng_encode32_file(output_filename, h_output_image, width, height);

	free(h_input_image);
	free(h_output_image);
}

cudaError_t processWithCuda(unsigned char * h_input_image, unsigned char * h_output_image, const int IMAGE_BYTES, const int IMAGE_SIZE) {
	cudaError_t cudaStatus;
	unsigned char * d_input_image = 0;
	unsigned char * d_output_image = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc(&d_input_image, IMAGE_BYTES);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_output_image, IMAGE_BYTES);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_input_image, h_input_image, IMAGE_BYTES, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	int block_size = 1024;
	rectify << <block_size, IMAGE_SIZE / block_size >> >(d_output_image, d_input_image);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "rectify launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rectify!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(h_output_image, d_output_image, IMAGE_BYTES, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_output_image);
	cudaFree(d_input_image);

	return cudaStatus;
}

int main(int argc, char *argv[]) {
	char *input_filename = argv[1];
	char *output_filename = argv[2];
	int num_threads = atoi(argv[3]);

	process(input_filename, output_filename, num_threads);

	return 0;
}
