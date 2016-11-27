#include "lib/lodepng.h"
#include <stdio.h>

__device__ unsigned char pickLargest(unsigned char j, unsigned char k, unsigned char l, unsigned char m)
{
    unsigned char largest = j;
    if (k > largest)
    {
        largest = k;
    }

    if (l > largest)
    {
        largest = l;
    }

    if (m > largest)
    {
        largest = m;
    }

    return largest;
}

__global__ void pool(unsigned char *output, cudaTextureObject_t texObj, int width, int height, int output_width, int output_height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(x < output_width && y < output_height))
        return;

    uchar4 current = tex2D<uchar4>(texObj, x, y);
    uchar4 right = tex2D<uchar4>(texObj, x + 1, y);
    uchar4 down = tex2D<uchar4>(texObj, x, y + 1);
    uchar4 downRight = tex2D<uchar4>(texObj, x + 1, y + 1);

    output[(y * output_width + x) * 4] = pickLargest(current.x, right.x, down.x, downRight.x);
    output[(y * output_width + x) * 4 + 1] = pickLargest(current.y, right.y, down.y, downRight.y);
    output[(y * output_width + x) * 4 + 2] = pickLargest(current.z, right.z, down.z, downRight.z);
    output[(y * output_width + x) * 4 + 3] = 255;
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
    dim3 dimGrid((outWidth + dimBlock.x - 1) / dimBlock.x, (outHeight + dimBlock.y - 1) / dimBlock.y);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray *cuArray;
    cudaStatus = cudaMallocArray(&cuArray, &channelDesc, inWidth, inHeight);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocArray failed! %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpyToArray(cuArray, 0, 0, input, INBYTES, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "input cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
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

    printf("Grid: %d %d %d, Block: %d %d %d", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    printf("In: %d %d, Out: %d %d", inWidth, inHeight, outWidth, outHeight);

    pool<<<dimGrid, dimBlock>>>(out, texObj, inWidth, inHeight, outWidth, outHeight);

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
        fprintf(stderr, "output cudaMemcpy failed! %s\n", cudaGetErrorString(cudaStatus));
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

    outWidth = inWidth / 2;
    outHeight = inHeight / 2;
    output = (unsigned char *)malloc(outWidth * outHeight * 4 * sizeof(unsigned char));

    processWithCuda(input, inWidth, inHeight, output, outWidth, outHeight);

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