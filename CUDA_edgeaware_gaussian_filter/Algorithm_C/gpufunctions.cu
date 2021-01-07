#include <iostream>
#include <thrust/complex.h>
#include "header.cuh"
#include <vector>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono> 

using namespace std;
using namespace std::chrono; 


//add vector function
__global__ void vecAdd(uchar4 *a, uchar4 *b, uchar4 *c, int n)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int aux;

    if (id < n){
        aux = a[id].x + b[id].x;
        c[id].x = aux;
        aux = a[id].y + b[id].y;
        c[id].y = aux;
        aux = a[id].z + b[id].z;
        c[id].z = aux;

    }
}

uchar4 *convert_uimg_to_uchar4(int width, int height, int channels, unsigned char *img){
    uchar4 *output;
    cudaMallocHost(&output,height* width* sizeof(uchar4));

    int i,j;

    for(i=0;i<height;i++){
        for(j=0;j<width;j++){
            output[i*width+j].x = img[i*width*channels+ j*channels];
            output[i*width+j].y = img[i*width*channels+ j*channels + 1];
            output[i*width+j].z = img[i*width*channels+ j*channels + 2];
            output[i*width+j].w = 255; //define opacity
        }
    }

    return output;
}

void transfer_uchar4_uint(int width, int height, int channels, uchar4 *input, unsigned char *output){
    int i,j;

    for(i = 0;i < height; i++){
        for(j=0;j<width;j++){
            output[i*width*channels + j*channels] = input[i*width+j].x;
            output[i*width*channels + j*channels+1] = input[i*width+j].y;
            output[i*width*channels + j*channels+2] = input[i*width+j].z;
        }
    }


}

void compute_constants(thrust::complex<float> * constant, float sigma_h){
    thrust::complex <float> sigma(sigma_h,0.00);
    //gamma
    constant[0] = alpha0*(unit+exp(-lambda0/sigma))/(unit-exp(-lambda0/sigma)) + alpha1*(unit+exp(-lambda1/sigma))/(unit-exp(-lambda1/sigma));
        //a0 a1
    constant[1] = alpha0/constant[0]; 
    constant[2] = alpha1/constant[0];
        //b0 b1
    constant[3] = exp(-lambda0/sigma);  
    constant[4] = exp(-lambda1/sigma);
        //r00 r01
    constant[5] = (constant[3]-unit)*(constant[3]-unit)/(constant[1]*constant[3]);  
    constant[6] = constant[1]/(constant[3]-unit);
        //r10 r11
    constant[7] = (constant[4]-unit)*(constant[4]-unit)/(constant[2]*constant[4]);  
    constant[8] = constant[2]/(constant[4]-unit);

    //theta b0 b1
    constant[9] = atan(constant[3].imag()/constant[3].real());
    constant[10] = atan(constant[4].imag()/constant[4].real());
    //radius b0 b1
    constant[11] = sqrtf(constant[3].real()*constant[3].real() + constant[3].imag()*constant[3].imag());
    constant[12] = sqrtf(constant[4].real()*constant[4].real() + constant[4].imag()*constant[4].imag());
}
void image_kernel_call_horizontal(uchar4 *auximage,uchar4 *outputimage,float sigma_h,float sigma_r,thrust::complex <float> *constant, uchar4  *img,int width, int height, int channels ,int window_w, int window_h, float kappa,int line_count){
    
    cudaStream_t stream1;
    cudaStream_t stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    int sharedpad=5;

    int nv_blocks = ceil(height/(1.00*window_h));

    dim3 grid(nv_blocks,line_count,1); 
    dim3 block(window_h,1,1); //height: number of cores used
    
    float s_quotient = sigma_h/sigma_r;
    s_quotient = s_quotient* s_quotient;

    gaussian_filter_kernel_horizontal_causal<<<grid,block,(window_h*(window_w+sharedpad))*sizeof(uchar4),stream1>>>(auximage,sigma_h,s_quotient,constant,img,width,  height, channels,window_w,window_h, sharedpad, kappa,line_count);
    gaussian_filter_kernel_horizontal_anticausal<<<grid,block,(window_h*(window_w+sharedpad))*sizeof(uchar4),stream2>>>(outputimage,sigma_h,s_quotient,constant,img,width,  height, channels,window_w,window_h,sharedpad, kappa,line_count);
    cudaDeviceSynchronize();
    dim3 grid2(ceil(width*height/window_h),1,1);
    dim3 block2(window_h,1,1);
    vecAdd<<<grid2,block2>>>(auximage,outputimage,outputimage,width*height);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
}

void image_kernel_call_vertical(uchar4 *auximage,uchar4 *outputimage,float sigma_h,float sigma_r,thrust::complex <float> *constant, uchar4  *img,int width, int height, int channels ,int window_w, int window_h, float kappa,int line_count){
    
    cudaStream_t stream1;
    cudaStream_t stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int sharedpad=5; 
    int aux = window_w;
    window_w = window_h;
    window_h = aux;

    int nh_blocks = ceil(width/(1.00*window_w));

    dim3 grid(nh_blocks,line_count,1); 
    dim3 block(window_w,1,1);
    
    float s_quotient = sigma_h/sigma_r;
    s_quotient = s_quotient* s_quotient;

    gaussian_filter_kernel_vertical_causal<<<grid,block,(window_h*(window_w+0))*sizeof(uchar4)>>>(outputimage,sigma_h,s_quotient,constant,img,width,  height, channels,window_w,window_h, 0,kappa,line_count);
    gaussian_filter_kernel_vertical_anticausal<<<grid,block,(window_h*(window_w+0))*sizeof(uchar4)>>>(auximage,sigma_h,s_quotient,constant,img,width,  height, channels,window_w,window_h, 0,kappa,line_count);
    
    cudaDeviceSynchronize();
    dim3 grid2(ceil(width*height/window_h),1,1);
    dim3 block2(window_h,1,1);
    vecAdd<<<grid2,block2>>>(auximage,outputimage,outputimage,width*height);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

}



void image_filter2d(float sigma_h,float sigma_r,uchar4 *inputimage,int width,int height, int channels,int window_w, int window_h, float kappa, int line_count){

    uchar4 *device_in, *device_out, *aux, *auximage;

    float sigma;
    float ratio = 1.00/2.00; // sigma is multiplied by "ratio"
    int num_it = 2,i;
   
    //COMPUTE CONSTANTS

    thrust::complex <float> constant[13];
    thrust::complex <float> *device_constant;


    cudaMalloc(&device_constant, 13*sizeof(thrust::complex<float>));

    cudaMalloc(&device_in,width*height*sizeof(uchar4));
    cudaMalloc(&auximage,width*height*sizeof(uchar4));
    cudaMalloc(&device_out,width*height*sizeof(uchar4));

    sigma = sigma_h*sqrtf((ratio*ratio-1)/(powf(ratio,2*(2*num_it))-1));

    cudaMemcpy(device_in, inputimage, width*height*sizeof(uchar4), cudaMemcpyHostToDevice);
    auto start = high_resolution_clock::now(); 
   

    
    for(i=0;i<num_it ;i++){

        compute_constants(constant,sigma);
        cudaMemcpy(device_constant, constant, 13*sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
        image_kernel_call_horizontal(auximage,device_out,sigma_h,sigma_r,device_constant,device_in,width,height,channels,window_w,window_h,kappa,line_count);
        aux = device_out;
        device_out = device_in;
        device_in = aux;
        sigma = sigma*ratio;


        compute_constants(constant,sigma);
        cudaMemcpy(device_constant, constant, 13*sizeof(thrust::complex<float>), cudaMemcpyHostToDevice);
        image_kernel_call_vertical(auximage,device_out,sigma_h,sigma_r,device_constant,device_in,width,height,channels,window_w,window_h,kappa,line_count);
        aux = device_out;
        device_out = device_in;
        device_in = aux;
        sigma = sigma*ratio;

    }


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);    
    float exectime = duration.count()/1000.00;
    printf("exectime %f\n",exectime);
    cudaMemcpy(inputimage, aux, width*height*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_constant);
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(auximage);
    cudaDeviceSynchronize();

}


