#include "interp2.cuh"
#include <iostream>
#include <vector>
#include <math.h>


__constant__ float kern_d[3];

int main() {



    const int size_row = 100;
    const int size_col = 100;

    std::vector<float> img((size_row +1) * (size_col + 1));

    const int sizeImg = sizeof(float) * size_row * size_col;
    const int sizeMem = sizeof(float) * (size_row + 1 ) * (size_col + 1);

    const int max_size = sizeof(float)*std::max(size_row, size_col);

    for(int i = 1; i < (int) size_row-1; i++)
    {
        img[i + size_row * ((int)size_col/2)] = 1;
    }

    // e-1                   e-2  e-1  e-2
    //  1    x  e-1 1 e-1 =  e-1   1   e-1
    // e-1                   e-2  e-1  e-2

    float sum = 2*expf(-1) + 1;
    float kenr_h[] = {expf(-1)/sum, (float)1.0/sum, expf(-1)/sum};
    return 0;
}
