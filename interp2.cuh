#ifndef INT2_INTERP2_CUH
#define INT2_INTERP2_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <list>
#include <vector>

#include "device_launch_parameters.h"

typedef  std::vector<float> img_t;

__device__ __constant__ float conv1[3];
__device__ __constant__ float conv2[3];

__global__ void CalculatingIn4Mem(float *mem1, float *mem2, float *mem3, float *mem4,
                                  const float *img, const unsigned int sizeRow,
                                  const unsigned int sizeCol)
{
    unsigned int idx = 1 + threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = 1 + threadIdx.y + blockDim.y * blockIdx.y;


    for (unsigned int i = idy; i < sizeRow; i += blockDim.y * gridDim.y)
    {
        for (unsigned int j = idx; j < sizeCol; j += blockDim.x * gridDim.x)
        {
            unsigned int img_id = i + (sizeRow + 1) * j;
            mem1[img_id - 1] += conv1[0] * img[img_id];
            mem2[img_id + 1 ] += conv1[2] * img[img_id];
            mem3[img_id - sizeRow] += conv2[0] * img[img_id];
            mem4[img_id + sizeRow] += conv2[2] * img[img_id];
        }
    }
}


__global__ void Sum(const float *mem1, const float *mem2, const float *mem3,
                    const float *mem4, float *img,const unsigned int sizeRow,
                    const unsigned int sizeCol)
{
    unsigned int idx = 1 + threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int idy = 1 + threadIdx.y + blockDim.y * blockIdx.y;

    for (unsigned int i = idy; i < sizeRow; i += blockDim.y * gridDim.y)
    {
        for (unsigned int j = idx; j < sizeCol; j += blockDim.x * gridDim.x)
        {
            unsigned int img_id = i + (sizeRow + 1) * j;
            img[img_id] +=  mem1[img_id - 1]
                            + mem2[img_id + 1]
                            + mem3[img_id - sizeRow]
                            + mem4[img_id + sizeRow]
                            + conv1[1] * img[img_id]
                            + conv2[2] * img[img_id];
        }
    }
}

class Conv2d
{
public:
    Conv2d(img_t &_img, const unsigned int &_sizeRow, const unsigned int &_sizeCol,
           const float *conv1_h, const float  *conv2_h) :
        img_data(_img.data()),
        mem1(nullptr),
        mem2(nullptr),
        mem3(nullptr),
        mem4(nullptr),
        sizeImg(_sizeRow * _sizeCol),
        sizeMem( (_sizeRow + 1) * (_sizeCol + 1) ),
        sizeCol(_sizeCol),
        sizeRow(_sizeRow)
    {

        AllocAllMem(conv1_h, conv2_h);
    }
    ~Conv2d()
    {
        cudaFree(conv1);
        cudaFree(conv2);
        cudaFree(d_img);
        cudaFree(mem1);
        cudaFree(mem2);
        cudaFree(mem3);
        cudaFree(mem4);
    }
    cudaError_t AllocAllMem(const float *conv1_h,
                            const float *conv2_h)
    {
        cudaMalloc((void**)&mem1, sizeMem);
        cudaMalloc((void**)&mem2, sizeMem);
        cudaMalloc((void**)&mem3, sizeMem);
        cudaMalloc((void**)&mem4, sizeMem);
        cudaMalloc((void**)&d_img, sizeImg);

        cudaMemcpy(d_img, img_data, sizeImg, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(conv1, &conv1_h, 3* sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(conv2, &conv2_h, 3* sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaError_t Smooth()
    {
        const unsigned int block_x = ceil(sizeCol/256);
        const unsigned int block_y = ceil(sizeRow/256);

        dim3 grid(block_x,block_y);
        dim3 thread(256,256);

        CalculatingIn4Mem<<<grid,thread>>>(mem1, mem2, mem3, mem4,
                                           d_img, sizeRow, sizeCol);

        Sum<<<grid,thread>>>(mem1, mem2, mem3, mem4,
                             d_img, sizeRow, sizeCol);
    }
    cudaError_t AttachToHost(img_t &img)
    {
        float *h_img_result = nullptr;
        cudaMemcpyToSymbol(h_img_result, &d_img, cudaMemcpyDeviceToHost);
        img.insert(img.end(), &h_img_result[0], &h_img_result[sizeRow*sizeCol-1]);
    }

private:
    float *img_data;
    float *d_img;
    float *mem1;
    float *mem2;
    float *mem3;
    float *mem4;
    const unsigned int sizeImg;
    const unsigned int sizeMem;
    const unsigned int sizeRow;
    const unsigned int sizeCol;
};
#endif //INT2_INTERP2_CUH
