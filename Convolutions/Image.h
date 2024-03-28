#include <cstdio>
#include <stdint.h>

struct Image {
    uint8_t* data = NULL;
    size_t size = 0;
    int w;
    int h;
    int channels;

    Image(const char* filename);
    Image(int w, int h, int channels);
    Image(Image& img);
    ~Image();

    bool read(const char* filename);
    bool write(const char* filename);
    void cuda_convolve(uint32_t ker_width, uint32_t ker_height, double ker[], uint32_t centre_r, uint32_t centre_c);
};