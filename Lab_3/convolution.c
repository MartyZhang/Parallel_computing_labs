#include "lib/lodepng.h"
#include <stdio.h>

__device__ unsigned char clamp(float c)
{
    if (c < 0)
        return 0;
    if (c > 255)
        return 255;
    return c;
}

__device__ float w[3][3] =
    {
        1, 2, -1,
        2, 0.25, -2,
        1, -2, -1};

__global__ void conv(unsigned char *output, cudaTextureObject_t texObj, int width, int height, int output_width, int output_height)
{
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(x < output_width && y < output_height))
        return;

    float r = 0, g = 0, b = 0;
    float u = 0;
    for (int ii = 0; ii < 3; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
        {
            uchar4 pixel = tex2D<uchar4>(texObj, x + ii, y + jj);
            u = w[ii][jj];
            r += pixel.x * u;
            g += pixel.y * u;
            b += pixel.z * u;
        }
    }

    output[(y * output_width + x) * 4] = clamp(r);
    output[(y * output_width + x) * 4 + 1] = clamp(g);
    output[(y * output_width + x) * 4 + 2] = clamp(b);
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
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.maxAnisotropy = 0;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
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

    printf("Grid: %d %d %d, Block: %d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    printf("In: %d %d, Out: %d %d\n", inWidth, inHeight, outWidth, outHeight);

    conv<<<dimGrid, dimBlock>>>(out, texObj, inWidth, inHeight, outWidth, outHeight);

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

    outWidth = inWidth - 2;
    outHeight = inHeight - 2;
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