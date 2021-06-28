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
            unsigned int mem_id = 1 + i + (sizeRow + 2) * (1 + j);
            mem1[mem_id - k] += conv[0 + n] * img[img_id];
            mem2[mem_id + k] += conv[2 + n] * img[img_id];
        }
    }
}

//__global__ void Sum(const float *mem1, const float *mem2,
//                    float *img, const unsigned int sizeRow,
//                    const unsigned int sizeCol, const unsigned  int k,
//                    const unsigned  int n)
//{
//    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;

//    for (unsigned int i = idy; i < sizeRow; i += blockDim.y * gridDim.y)
//    {
//        for (unsigned int j = idx; j < sizeCol; j += blockDim.x * gridDim.x)
//        {
//            unsigned int img_id = i + (sizeRow) * j;
//            unsigned int mem_id = 1 + i + (sizeRow + 2) * (1 + j);
//            img[img_id] +=  mem1[mem_id - k]
//                            + mem2[mem_id + k]
//                            + conv[1 + n] * img[img_id];
//        }
//    }
//}


//__global__ void CalculatingSmoothParts(float *mem1, float *mem2,
//                                       const float *img, const unsigned int sizeRow,
//                                       const unsigned int sizeCol, const unsigned  int k,
//                                       const unsigned  int n)
//{
//    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    unsigned int idy = threadIdx.y + blockDim.y * blockIdx.y;


//    for (unsigned int i = idy; i < sizeRow; i += blockDim.y * gridDim.y)
//    {
//        for (unsigned int j = idx; j < sizeCol; j += blockDim.x * gridDim.x)
//        {
//            unsigned int img_id = i + (sizeRow) * j;
//            unsigned int mem_id = 1 + i + (sizeRow + 2) * (1 + j);
//            mem1[mem_id - k] += conv[0 + n] * img[img_id];
//            mem2[mem_id + k] += conv[2 + n] * img[img_id];
//        }
//    }
//}

void Suma(const float *mem1, const float *mem2,
                    float *img, const unsigned int sizeRow,
                    const unsigned int sizeCol, const unsigned  int k,
                    const unsigned  int n,
          const float *conv)
{

    for (unsigned int i = 0; i < sizeRow; i += 1)
    {
        for (unsigned int j = 0; j < sizeCol; j += 1)
        {
            unsigned int img_id = i + (sizeRow) * j;
            unsigned int mem_id = 1 + i + (sizeRow + 2) * (1 + j);
            img[img_id] +=  mem1[mem_id - k]
                            + mem2[mem_id + k]
                            + conv[1 + n] * img[img_id];
        }
    }
}

void CalculatingSmoothParts1(float *mem1, float *mem2,
                                       const float *img, const unsigned int sizeRow,
                                       const unsigned int sizeCol, const unsigned  int k,
                                       const unsigned  int n,
                             const float *conv)
{

    for (unsigned int i = 0; i < sizeRow; i += 1)
    {
        for (unsigned int j = 0; j < sizeCol; j += 1)
        {
            unsigned int img_id = i + (sizeRow) * j;
            unsigned int mem_id = 1 + i + (sizeRow + 2) * (1 + j);
            mem1[mem_id - k] += conv[0 + n] * img[img_id];
            mem2[mem_id + k] += conv[2 + n] * img[img_id];
        }
    }
}

void cudaCheck(cudaError_t err)
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
        mem1((float*)malloc(sizeMem)),
        mem2((float*)malloc(sizeMem)),
        sizeRow(_sizeRow),
        sizeCol(_sizeCol)
    {

        cudaCheck(AllocAllMem(conv_h));
    }
    ~Conv2d()
    {
//        cudaFree(conv);
        cudaFree(d_img);
        cudaFree(mem1);
        cudaFree(mem2);
    }
    cudaError_t AllocAllMem(const float *conv_h)
    {
//        cudaCheck(cudaMalloc((void**)&mem1, sizeMem));
//        cudaCheck(cudaMalloc((void**)&mem2, sizeMem));
//        cudaCheck(cudaMalloc((void**)&d_img, sizeImg));
        float *mem1 = (float*)malloc(sizeMem);
        float *mem2 = (float*)malloc(sizeMem);
//        conv = conv_h;
//        cudaCheck(cudaMemcpy(d_img, img_data, sizeImg, cudaMemcpyHostToDevice));
//        cudaCheck(cudaMemcpyToSymbol(conv, conv_h, 6* sizeof(float)));

        return cudaGetLastError();
    }
    cudaError_t Smooth(const float *conv_h)
    {
        const unsigned int block_x = ceil((float)sizeCol/16);
        const unsigned int block_y = ceil((float)sizeRow/16);

        dim3 grid(block_x,block_y);
        dim3 thread(16,16);

        std::vector<float> vec;
        CalculatingSmoothParts1(mem1, mem2, img_data, sizeRow, sizeCol, 1, 0,conv_h);
        //cudaCheck(cudaGetLastError());

        Suma(mem1, mem2,img_data, sizeRow, sizeCol, 1, 0,conv_h);
         AttachToHostMem(vec);
        cudaCheck(cudaGetLastError());
        CalculatingSmoothParts1(mem1, mem2, img_data, sizeRow, sizeCol, sizeRow, 3,conv_h);
        cudaCheck(cudaGetLastError());
        Suma(mem1, mem2,img_data, sizeRow, sizeCol, sizeRow, 3,conv_h);
        cudaCheck(cudaGetLastError());
        return cudaGetLastError();
    }
    cudaError_t AttachToHost(img_t &img)
    {
        float *h_img_result = (float*)malloc(sizeImg);
        cudaMemcpy(h_img_result, d_img, sizeImg, cudaMemcpyDeviceToHost);
        img.insert(img.end(), h_img_result, h_img_result + sizeRow*sizeCol);
        return cudaGetLastError();
    }
    cudaError_t AttachToHostMem(img_t &img)
    {
        float *h_img_result = (float*)malloc(sizeMem);
        cudaMemcpy(h_img_result, mem1, sizeMem, cudaMemcpyDeviceToHost);
        img.insert(img.end(), h_img_result, h_img_result + (1+sizeRow)*(1+sizeCol));
        return cudaGetLastError();
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
