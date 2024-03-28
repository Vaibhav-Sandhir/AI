#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define BYTE_BOUND(value) value < 0 ? 0 : (value > 255 ? 255 : value)
#include "Image.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Image::Image(const char* filename){
    if(read(filename)){
        printf("\nRead %s", filename);
        size = w*h*channels;
    }
    else{
        printf("\nFailed to read %s", filename); 
    }
}

Image::Image(int w, int h, int channels) : w(w), h(h), channels(channels){
    size = w*h*channels;
    data = new uint8_t[size];
}

Image::Image(Image& img) : Image(img.w, img.h, img.channels){
    memcpy(data, img.data, img.size);
}

Image::~Image(){
    stbi_image_free(data);
}

bool Image::read(const char* filename){
    data = stbi_load(filename, &w, &h, &channels, 0);
    return data != NULL;
}

bool Image::write(const char* filename){
    int success;
    success = stbi_write_png(filename, w, h, channels, data, w*channels);
    return success != 0;
}

// Image& Image::std_convolve(uint32_t ker_width, uint32_t ker_height, double ker[], uint32_t centre_r, uint32_t centre_c) {
// 	uint8_t new_data[w*h];
// 	uint64_t center = centre_r*ker_width + centre_c;
//     for(int channel = 0; channel < 3; channel++){
//         for(uint64_t k=channel; k<size; k+=channels) {
//             double c = 0;
//             for(long i = -((long)centre_r); i<(long)ker_height-centre_r; ++i) {
//                 long row = ((long)k/channels)/w-i;
//                 if(row < 0 || row > h-1) {
//                     continue;
//                 }
//                 for(long j = -((long)centre_c); j<(long)ker_width-centre_c; ++j) {
//                     long col = ((long)k/channels)%w-j;
//                     if(col < 0 || col > w-1) {
//                         continue;
//                     }
//                     c += ker[center+i*(long)ker_width+j]*data[(row*w+col)*channels+channel];
//                 }
//             }
//             new_data[k/channels] = (uint8_t)BYTE_BOUND(round(c));
//         }
//         for(uint64_t k=channel; k<size; k+=channels) {
// 		    data[k] = new_data[k/channels];
// 	    }
//     }    
// 	return *this;
// }

    __global__ void cuda_convolution_kernel(uint8_t* data, int w, int h, int channels, uint32_t ker_width, uint32_t ker_height, double* ker, uint32_t centre_r, uint32_t centre_c) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (col < w && row < h) {
            uint64_t center = centre_r * ker_width + centre_c;
            for (int channel = 0; channel < channels; ++channel) {
                double c = 0;
                for (long i = -((long)centre_r); i < (long)ker_height - centre_r; ++i) {
                    long r = row - i;
                    if (r < 0 || r >= h) {
                        continue;
                    }
                    for (long j = -((long)centre_c); j < (long)ker_width - centre_c; ++j) {
                        long c_idx = col - j;
                        if (c_idx < 0 || c_idx >= w) {
                            continue;
                        }
                        double kernel_val = ker[center + i * ker_width + j];
                        c += kernel_val * data[(r * w + c_idx) * channels + channel];
                    }
                }
                data[(row * w + col) * channels + channel] = (uint8_t)BYTE_BOUND(round(c));
            }
        }
    }

    void Image::cuda_convolve(uint32_t ker_width, uint32_t ker_height, double* ker, uint32_t centre_r, uint32_t centre_c) {
        uint8_t* cuda_data;
        double* cuda_ker;
        cudaMalloc(&cuda_data, size * sizeof(uint8_t));
        cudaMalloc(&cuda_ker, ker_width * ker_height * sizeof(double));
        cudaMemcpy(cuda_data, data, size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_ker, ker, ker_width * ker_height * sizeof(double), cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
        cuda_convolution_kernel<<<gridSize, blockSize>>>(cuda_data, w, h, channels, ker_width, ker_height, cuda_ker, centre_r, centre_c);

        cudaMemcpy(data, cuda_data, size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        cudaFree(cuda_data);
        cudaFree(cuda_ker);
    }
