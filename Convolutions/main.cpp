#include "Image.h"

int main(int argc, char** argv){
    Image test("b.png");
    double ker[] = {2/16.0, 4/16.0, 2/16.0,
                     4/16.0, 8/16.0, 4/16.0,
                     2/16.0, 4/16.0, 2/16.0};               
    test.cuda_convolve(3, 3, ker, 1, 1);               
    test.write("highcontrast.png");
    return 0;
}