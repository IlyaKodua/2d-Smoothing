#ifndef INT2_INTERP2_CUH
#define INT2_INTERP2_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <list>
#include <vector>

#include "device_launch_parameters.h"

typedef  std::vector<float> img_t;

__device__ __constant__ float conv[6];

__global__ void CalculatingSmoothParts(float *mem1, float *mem2,
                                       const float *img, const unsigned int sizeRow,
                                       const unsigned int sizeCol, const unsigned  int k,
                                       const unsigned  int n)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;


    for (unsigned int i = idy; i < sizeRow; i += blockDim.y * gridDim.y)
    {
        for (unsigned int j = idx; j < sizeCol; j += blockDim.x * gridDim.x)
        {
            unsigned int img_id = i + (sizeRow) * j;
            unsigned int mem_id = 1 + i * (sizeRow+1) * j;
            mem1[mem_id - k] += conv[0 + n] * img[img_id];
            mem2[mem_id + k] += conv[2 + n] * img[img_id];
        }
    }
}

__global__ void Sum(const float *mem1, const float *mem2,
                    float *img, const unsigned int sizeRow,
                    const unsigned int sizeCol, const unsigned  int k,
                    const unsigned  int n)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;

    for (unsigned int i = idy; i < sizeRow; i += blockDim.y * gridDim.y)
    {
        for (unsigned int j = idx; j < sizeCol; j += blockDim.x * gridDim.x)
        {
            unsigned int img_id = i + (sizeRow) * j;
            unsigned int mem_id = 1 + i * (sizeRow+1) * j;
            img[img_id] +=  mem1[img_id - k]
                            + mem2[mem_id + k]
                            + conv[1 + n] * img[img_id];
        }
    }
}

class Conv2d
{
public:
    Conv2d(img_t &_img, const unsigned int &_sizeRow, const unsigned int &_sizeCol,
           const float *conv_h) :
        img_data(_img.data()),
        mem1(nullptr),
        mem2(nullptr),
        sizeImg(_sizeRow * _sizeCol),
        sizeMem( (_sizeRow + 1) * (_sizeCol + 1) ),
        sizeCol(_sizeCol),
        sizeRow(_sizeRow)
    {

        AllocAllMem(conv_h);
    }
    ~Conv2d()
    {
        cudaFree(conv);
        cudaFree(d_img);
        cudaFree(mem1);
        cudaFree(mem2);
    }
    cudaError_t AllocAllMem(const float *conv_h)
    {
        cudaMalloc((void**)&mem1, sizeMem);
        cudaMalloc((void**)&mem2, sizeMem);
        cudaMalloc((void**)&d_img, sizeImg);

        cudaMemcpy(d_img, img_data, sizeImg, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(conv, &conv_h, 6* sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaError_t Smooth()
    {
        const unsigned int block_x = ceil(sizeCol/256);
        const unsigned int block_y = ceil(sizeRow/256);

        dim3 grid(block_x,block_y);
        dim3 thread(256,256);

        CalculatingSmoothParts<<<grid,thread>>>(mem1, mem2, d_img, sizeRow, sizeCol, 1, 0);
        Sum<<<grid,thread>>>(mem1, mem2,d_img, sizeRow, sizeCol, 1, 0);
        CalculatingSmoothParts<<<grid,thread>>>(mem1, mem2, d_img, sizeRow, sizeCol, sizeRow, 3);
        Sum<<<grid,thread>>>(mem1, mem2,d_img, sizeRow, sizeCol, sizeRow, 3);
    }
    cudaError_t AttachToHost(img_t &img)
    {
        float *h_img_result = nullptr;
        cudaMemcpy(h_img_result, &d_img,sizeImg, cudaMemcpyDeviceToHost);
        img.insert(img.end(), &h_img_result[0], &h_img_result[sizeRow*sizeCol]);
    }

private:
    float *img_data;
    float *d_img;
    float *mem1;
    float *mem2;
    const unsigned int sizeImg;
    const unsigned int sizeMem;
    const unsigned int sizeRow;
    const unsigned int sizeCol;
};
#endif //INT2_INTERP2_CUH
