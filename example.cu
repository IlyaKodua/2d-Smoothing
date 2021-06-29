#include "conv2d.cuh"
#include <vector>

int main() {



    const int size_row = 2;
    const int size_col = 2;

    std::vector<float> img((size_row) * (size_col));

    img[1] = 1;
    img[2] = 1;

    // 0.5                   0.25  0.5  0.25
    //  1   x 0.5  1 0.5  =  0.5    1   0.5
    // 0.5                   0.25  0.5  0.25


    float conv_h[] = {0.5, 1.0, 0.5,// conv1
                      0.5, 1.0, 0.5};// conv2

    Conv2d conv(img, size_row, size_col, conv_h);
    conv.Smooth();
    img_t vec;
    conv.AttachToHost(vec);
    return 0;
}
