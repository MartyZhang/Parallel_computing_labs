#include "lib/lodepng.h"
#include <stdio.h>

__global__ void rectify(unsigned char *output, cudaTextureObject_t texObj, int width, int height, int output_width, int output_height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(x < width && y < height))
	return;

    uchar4 pixel = tex2D<uchar4>(texObj, x, y);
    output[(y * width + x) * 4] = pixel.x < 127 ? 127 : pixel.x;
    output[(y * width + x) * 4 + 1] = pixel.y < 127 ? 127 : pixel.y;
    output[(y * width + x) * 4 + 2] = pixel.z < 127 ? 127 : pixel.z;
    output[(y * width + x) * 4 + 3] = 255;
}

//cudaError_t processWithCuda(Image *input, Image *output)
cudaError_t processWithCuda(unsigned char *input, int inWidth, int inHeight, unsigned char *output, int outWidth, int outHeight)
{
    cudaError_t cudaStatus;
    unsigned char *out;
    const int INBYTES = (inWidth * inHeight * 4 * sizeof(unsigned char));
    const int OUTBYTES = (outWidth * outHeight * 4 * sizeof(unsigned char));

    // Invoke kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((inWidth + dimBlock.x - 1) / dimBlock.x, (inHeight + dimBlock.y - 1) / dimBlock.y);
    
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray *cuArray;
    cudaStatus = cudaMallocArray(&cuArray, &channelDesc, inWidth, inHeight);
    if (cudaStatus != cudaSuccess)
    {
	fprintf(stderr, "cudaMallocArray failed!");
	return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpyToArray(cuArray, 0, 0, input, INBYTES, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
	fprintf(stderr, "cudaMemcpy failed!");
	return cudaStatus;
    }

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Set texture reference parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    cudaStatus = cudaMalloc(&out, OUTBYTES);
    if (cudaStatus != cudaSuccess)
    {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
    }

    rectify<<<dimGrid, dimBlock>>>(out, texObj, inWidth, inHeight, outWidth, outHeight);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
	fprintf(stderr, "rectify launch failed: %s\n", cudaGetErrorString(cudaStatus));
	goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, out, OUTBYTES, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
    }

Error:
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(out);

    return cudaStatus;
}

void process(char *input_filename, char *output_filename)
{
    unsigned error = 0;
    unsigned char *input, *output;
    unsigned inWidth, inHeight, outWidth, outHeight;

    if (lodepng_decode32_file(&input, &inWidth, &inHeight, input_filename))
	printf("error %u: %s\n", error, lodepng_error_text(error));

    outWidth = inWidth;
    outHeight = inHeight;
    output = (unsigned char *)malloc(outWidth * outHeight * 4 * sizeof(unsigned char));

    cudaError_t cudaStatus = processWithCuda(input, inWidth, inHeight, output, outWidth, outHeight);
    if (cudaStatus != cudaSuccess)
    {
	fprintf(stderr, "failed!");
    }

    lodepng_encode32_file(output_filename, output, outWidth, outHeight);

    free(input);
    free(output);
}

int main(int argc, char *argv[])
{
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    process(input_filename, output_filename);
    return 0;
}