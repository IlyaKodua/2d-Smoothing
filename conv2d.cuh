#ifndef INT2_INTERP2_CUH
#define INT2_INTERP2_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <list>
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include "device_launch_parameters.h"

typedef  std::vector<float> img_t;

__constant__ float conv[6];


__global__ void sum(const float *mem1, const float *mem2,
                    float *img, const unsigned int sizeRow,
                    const unsigned int sizeCol,
                    const unsigned  int n)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;

    for (unsigned int i = idy; i < sizeRow; i += blockDim.y * gridDim.y)
    {
        for (unsigned int j = idx; j < sizeCol; j += blockDim.x * gridDim.x)
        {
            unsigned int img_id = i + (sizeRow) * j;
            unsigned int mem_id = 1 + i + (sizeRow + 2) * (1 + j);
            img[img_id] = mem1[mem_id]
                          + mem2[mem_id]
                          + conv[1 + n] * img[img_id];
        }
    }
}


__global__ void calc2mem(float *mem1, float *mem2,
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
            unsigned int mem_id = 1 + i + (sizeRow + 2) * (1 + j);
            mem1[mem_id - k] = conv[0 + n] * img[img_id];
            mem2[mem_id + k] = conv[2 + n] * img[img_id];
        }
    }
}

void cudaCheckDebug(cudaError_t err)
{
    assert(err == cudaSuccess);
}
class Conv2d
{
public:
    Conv2d(img_t &_img, const unsigned int &_sizeRow, const unsigned int &_sizeCol,
           const float *conv_h) :
        img_data(_img.data()),
        sizeMem( sizeof(float)*(_sizeRow + 2) * (_sizeCol + 2) ),
        sizeImg(sizeof(float)*_sizeRow * _sizeCol),
        sizeRow(_sizeRow),
        sizeCol(_sizeCol)
    {

        AllocAllMem(conv_h);
    }
    ~Conv2d()
    {
        cudaCheckDebug(cudaFree(d_img));
        cudaCheckDebug(cudaFree(mem1));
        cudaCheckDebug(cudaFree(mem2));
    }

    void AllocAllMem(const float *conv_h)
    {
        cudaCheckDebug(cudaMalloc((void**)&mem1, sizeMem));
        cudaCheckDebug(cudaMalloc((void**)&mem2, sizeMem));
        cudaCheckDebug(cudaMalloc((void**)&d_img, sizeImg));
        cudaCheckDebug(cudaMemcpy(d_img, img_data, sizeImg, cudaMemcpyHostToDevice));
        cudaCheckDebug(cudaMemcpyToSymbol(conv, conv_h, 6* sizeof(float)));
    }
    void Smooth()
    {
        const unsigned int block_x = ceil((float)sizeCol/16);
        const unsigned int block_y = ceil((float)sizeRow/16);

        dim3 grid(block_x,block_y);
        dim3 thrd(16,16);

        std::vector<float> vec;
        calc2mem<<<grid, thrd>>>(mem1, mem2, d_img, sizeRow, sizeCol, 1, 0);
        cudaCheckDebug(cudaGetLastError());

        sum<<<grid, thrd>>>(mem1, mem2, d_img, sizeRow, sizeCol, 0);
        cudaCheckDebug(cudaGetLastError());

        calc2mem<<<grid, thrd>>>(mem1, mem2, d_img, sizeRow, sizeCol, sizeRow+2, 3);
        cudaCheckDebug(cudaGetLastError());

        sum<<<grid, thrd>>>(mem1, mem2, d_img, sizeRow, sizeCol, 3);
        cudaCheckDebug(cudaGetLastError());
    }
    void AttachToHost(img_t &img)
    {
        img.resize(sizeImg/sizeof (float));
        cudaCheckDebug(cudaMemcpy(img.data(), d_img, sizeImg, cudaMemcpyDeviceToHost));
    }

private:
    float *img_data;
    const unsigned int sizeMem;
    const unsigned int sizeImg;
    float *d_img;
    float *mem1;
    float *mem2;
    const unsigned int sizeRow;
    const unsigned int sizeCol;
};
#endif //INT2_INTERP2_CUH
