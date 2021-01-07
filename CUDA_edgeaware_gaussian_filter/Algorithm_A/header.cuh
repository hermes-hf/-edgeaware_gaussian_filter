#include <iostream>
#include <thrust/complex.h>
#include <cuda_runtime.h>



using namespace std;

#ifndef HEADER_H

#define HEADER_H

    //Complex values table

    static const thrust::complex <float> alpha0(1.6800,3.7350);
    static const thrust::complex <float> alpha1(-0.6803,-0.2598);
    static const thrust::complex <float> lambda0(1.783,0.6318);
    static const thrust::complex <float> lambda1(1.723,1.9970);
    static const thrust::complex <float> unit(1.00,0.00);


    void image_filter2d(float *exectime,float sigma_r,float sigma_h,uchar4  *inputimage,int width,int height, int channels,int window_w, int window_h);

    __global__ 
    void gaussian_filter_kernel_horizontal_causal(uchar4 *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4 *img,int width, int height,int channels, int window_w, int window_h, int sharedpad);
    
    __global__
    void gaussian_filter_kernel_horizontal_anticausal(uchar4  *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4 *img,int width, int height,int channels, int window_w, int window_h, int sharedpad);

    __global__ 
    void gaussian_filter_kernel_vertical_causal(uchar4 *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4 *img,int width, int height,int channels, int window_w, int window_h, int sharedpad);
    
    __global__
    void gaussian_filter_kernel_vertical_anticausal(uchar4 *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4 *img,int width, int height,int channels, int window_w, int window_h, int sharedpad);


    void compute_constants(thrust::complex<float> * constant, float sigma_h);

    uchar4 *convert_uimg_to_uchar4(int width, int height, int channels, unsigned char *img);
    void transfer_uchar4_uint(int width, int height, int channels, uchar4 *input, unsigned char *output);


#endif
