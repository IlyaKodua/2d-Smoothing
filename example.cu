#include "interp2.cuh"
#include <iostream>
#include <vector>
#include <math.h>

int main() {

    __constant__ float kern_d[3];

    const int size_row = 100;
    const int size_col = 100;

    const int sizeImg = sizeof(float) * size_col * size_row;

    std::vector<float> img(size_row * size_col);

    for(int i = 0; i < (int) size_row; i++)
    {
        img[i + size_row * ((int)size_col/2)] = 1;
    }

    // e-1                   e-2  e-1  e-2
    //  1    x  e-1 1 e-1 =  e-1   1   e-1
    // e-1                   e-2  e-1  e-2

    float sum = 2*expf(-1) + 1;
    float kenr_h[] = {expf(-1)/sum, (float)1.0/sum, expf(-1)/sum};

    float *mem1;
    float *mem2;
    float *mem3;

    cudaMalloc((void**)&mem1, sizeImg);
    cudaMalloc((void**)&mem2, sizeImg);
    cudaMalloc((void**)&mem3, sizeImg);

    cudaMemcpyToSymbol(kern_d, &kenr_h, sizeImg, cudaMemcpyHostToDevice);


    return 0;
}
