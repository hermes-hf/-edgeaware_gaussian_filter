#include <iostream>
#include <thrust/complex.h>
#include "header.cuh"
#include <vector>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>


using namespace std;

__device__ void read_blockint_to_shareint_v(uchar4 *simg, uchar4 *img, int i_start, int i_end, int col_index, int block_col_index, int width, int channels, int window_w, int pad){
    int i;
    for(i=0;i<i_end-i_start;i++){
        simg[i*(window_w+pad) + block_col_index]=img[(i_start+i)*(width) + col_index];
    }
}

__device__ void write_shareint_to_block_v( uchar4 *simg, uchar4 *img, int i_start, int i_end, int col_index, int block_col_index, int width, int channels, int window_w,int pad){
    int i;

    for(i=0;i<i_end-i_start;i++){
        img[(i_start+i)*(width) + col_index]=simg[i*(window_w+pad) + block_col_index];
    }
}




__global__ 
void gaussian_filter_kernel_vertical_causal(uchar4 *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4 *img,int width, int height,int channels, int window_w, int window_h, int sharedpad,float kappa){

    int i,j,k,i_start=0,i_end = window_h, j_start,j_end;
    int block_i;
    uchar4 buffer;
    float dist;
    float f[3],f_prev[3],delta;
    extern __shared__ uchar4 simg[]; 
    thrust::complex<float> prevg0_causal[3],prevg0_acausal[3],prevg1_acausal[3],prevg1_causal[3],g0[3],g1[3];
    thrust::complex<float> b_delta,aux;

    j = threadIdx.x + window_w*blockIdx.x;
    
    while(i_start < height){
        aux.real(0);
        aux.imag(0);
        i_end = i_start +window_h;
        
        if(i_end> height){
            i_end = height;
        }

        i = i_start + threadIdx.x;
        j_start = blockIdx.x*window_w;
        j_end = j_start + window_w;
        if(j_end>width){j_end = width;}



        if(threadIdx.x<j_end-j_start)
        read_blockint_to_shareint_v(simg, img,i_start,i_end,  j_start+threadIdx.x, threadIdx.x, width,  channels, window_w, sharedpad);
        __syncthreads();


        j = threadIdx.x + window_w*blockIdx.x;
        
        if(1==1 && threadIdx.x<j_end-j_start){ 

            block_i = 0;
            if(kappa>=0){
                //APPROXIMATE
                dist=0;
                i = i_start;

                buffer = img[i*width + j];
                f_prev[0] = buffer.x;
                f_prev[1] = buffer.y;
                f_prev[2] = buffer.z;


                while(i>0 && (dist<sigma_h*kappa)){
                    i=i-1;
                    delta=0;
                    buffer = img[i*width +j];
                    f[0] = buffer.x;
                    f[1] = buffer.y;
                    f[2] = buffer.z;

                    for(k=0;k<channels;k++){
                        delta = delta + 1.00*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
                    }

                    delta = delta*s_quotient +1;
                    delta = sqrt(delta);
    
                    dist = dist + delta;
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                    }
                }
                // COMPUTE INITIAL CONDITIONS 
                dist=i;
                for(k=0;k<channels;k++){
                    f_prev[k] = f[k];
                    prevg0_causal[k] = constant[1]/(float(1.00)-constant[3])*f[k];
                    prevg1_causal[k] = constant[2]/(float(1.00)-constant[4])*f[k];
                }
                
                for(i = i; i< i_start; i++){
                    delta=0;
                    buffer = img[i*width + j];
                    f[0] = buffer.x;
                    f[1] = buffer.y;
                    f[2] = buffer.z;

                    if(i==dist){
                        for(k=0;k<channels;k++){
                            f_prev[k] = f[k];
                            prevg0_causal[k] = constant[1]/(float(1.00)-constant[3])*f[k];
                            prevg1_causal[k] = constant[2]/(float(1.00)-constant[4])*f[k];
                        }
                    }
    
                    for(k=0;k<channels;k++){
                        delta = delta + float(1.00)*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
                    }
    
                    delta = delta*s_quotient +1;
                    delta = sqrt(delta);
    
                    b_delta.real(cosf(delta*constant[9].real()));
                    b_delta.imag(sinf(delta*constant[9].real()));
                    b_delta = b_delta*powf(constant[11].real(),delta);
                    
    
                    aux = (b_delta-float(1.00))/(constant[5]*delta);
                    for(k = 0;k <channels; k++){
                        g0[k] = constant[1]*f[k]+ b_delta*prevg0_causal[k];
                        g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
                    }
    
                    b_delta.real(cosf(delta*constant[10].real()));
                    b_delta.imag(sinf(delta*constant[10].real()));
                    b_delta = b_delta*powf(constant[12].real(),delta);
    
                    aux= (b_delta-float(1.00))/(constant[7]*delta);
                    for(k = 0;k <channels; k++){
                        g1[k] = constant[2]*f[k]+ b_delta*prevg1_causal[k];
                        g1[k] = g1[k] + (aux - constant[8]*constant[4])*f[k] - (aux - constant[8]*b_delta)*f_prev[k];
                    }
        
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                        prevg0_causal[k] = g0[k];
                        prevg1_causal[k] = g1[k];
                    }
                }




            }


            for(i = i_start; i< i_end; i++){
                delta=0;
                buffer = simg[block_i*(window_w+ sharedpad) + threadIdx.x];
                f[0] = buffer.x;
                f[1] = buffer.y;
                f[2] = buffer.z;

                if(i==0){
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                        prevg0_causal[k] = constant[1]/(float(1.00)-constant[3])*f[k];
                        prevg1_causal[k] = constant[2]/(float(1.00)-constant[4])*f[k];
                    }
                }

                for(k=0;k<channels;k++){
                    delta = delta + float(1.00)*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
                }

                delta = delta*s_quotient +1;
                delta = sqrt(delta);

                b_delta.real(cosf(delta*constant[9].real()));
                b_delta.imag(sinf(delta*constant[9].real()));
                b_delta = b_delta*powf(constant[11].real(),delta);
                

                aux = (b_delta-float(1.00))/(constant[5]*delta);
                for(k = 0;k <channels; k++){
                    g0[k] = constant[1]*f[k]+ b_delta*prevg0_causal[k];
                    g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
                }

                b_delta.real(cosf(delta*constant[10].real()));
                b_delta.imag(sinf(delta*constant[10].real()));
                b_delta = b_delta*powf(constant[12].real(),delta);

                aux= (b_delta-float(1.00))/(constant[7]*delta);
                for(k = 0;k <channels; k++){
                    g1[k] = constant[2]*f[k]+ b_delta*prevg1_causal[k];
                    g1[k] = g1[k] + (aux - constant[8]*constant[4])*f[k] - (aux - constant[8]*b_delta)*f_prev[k];
                }
    
                for(k=0;k<channels;k++){
                    f_prev[k] = f[k];
                    prevg0_causal[k] = g0[k];
                    prevg1_causal[k] = g1[k];
                }
                
                aux.real(int(abs((g0[0]+g1[0]).real())));
                if(aux.real()>255){
                    buffer.x = 255;
                }
                else{ buffer.x = aux.real();}

                aux.real(int(abs((g0[1]+g1[1]).real())));
                if(aux.real()>255){
                    buffer.y = 255;
                }
                else{ buffer.y = aux.real();}

                aux.real(int(abs((g0[2]+g1[2]).real())));
                if(aux.real()>255){
                    buffer.z = 255;
                }
                else{ buffer.z = aux.real();}

                simg[block_i*(window_w+ sharedpad) + threadIdx.x] = buffer;

                block_i++;
            }
            
        }

        
        __syncthreads();
        if(threadIdx.x<j_end-j_start)
        write_shareint_to_block_v(simg, outputimage,i_start,i_end,  j_start+threadIdx.x, threadIdx.x, width,  channels, window_w, sharedpad);
        __syncthreads();



        i_start = i_end;
    }
}



__global__ 
void gaussian_filter_kernel_vertical_anticausal(uchar4 *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4 *img,int width, int height,int channels, int window_w, int window_h, int sharedpad, float kappa){

    int i,j,k,i_start=0,i_end = window_h, j_start,j_end;
    int block_i;
    uchar4 buffer;
    float dist;
    float f[3],f_prev[3],delta;
    extern __shared__ uchar4 simg[]; 
    thrust::complex<float> prevg0_acausal[3],prevg1_acausal[3],g0[3],g1[3];
    thrust::complex<float> b_delta,aux;

    j = threadIdx.x + window_w*blockIdx.x;
    i_end = height;
    i_start = i_end - window_h;

    while(i_end > 0){ 
        aux.real(0);
        aux.imag(0);
  
        i = i_start + threadIdx.x;
        j_start = blockIdx.x*window_w;
        j_end = j_start + window_w;
        if(j_end>width){j_end = width;}



        if(threadIdx.x<j_end-j_start)
        read_blockint_to_shareint_v(simg, img,i_start,i_end,  j_start+threadIdx.x, threadIdx.x, width,  channels, window_w, sharedpad);
        __syncthreads();


        j = threadIdx.x + window_w*blockIdx.x;
        



        if(threadIdx.x<j_end-j_start){ 

            block_i = i_end-i_start-1;
            
            if(kappa>=0){
                //APPROXIMATE
                dist=0;
                i = i_end-1;

                buffer = img[i*width + j];
                f_prev[0] = buffer.x;
                f_prev[1] = buffer.y;
                f_prev[2] = buffer.z;

                while(i<height-1 && (dist<sigma_h*kappa)){
                    i++;
                    delta=0;
                    buffer = img[i*width +j];
                    f[0] = buffer.x;
                    f[1] = buffer.y;
                    f[2] = buffer.z;

                    for(k=0;k<channels;k++){
                        delta = delta + 1.00*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
        
                    }
                    delta = delta*s_quotient +1;
                    delta = sqrt(delta);
    
                    dist = dist + delta;
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                    }
                }
                dist=i;
                for(k=0;k<channels;k++){
                    f_prev[k] = f[k];
                    prevg0_acausal[k] = constant[1]*constant[3]/(float(1.00)-constant[3])*f[k];
                    prevg1_acausal[k] = constant[2]*constant[4]/(float(1.00)-constant[4])*f[k];
                }
                for(i = i; i>= i_end; i=i-1){
                    delta=0;
                    buffer = img[i*width +j];
                    f[0] = buffer.x;
                    f[1] = buffer.y;
                    f[2] = buffer.z;

                    if(i==height-1){
                        for(k=0;k<channels;k++){
                            f_prev[k] = f[k];
                            prevg0_acausal[k] = constant[1]*constant[3]/(float(1.00)-constant[3])*f[k];
                            prevg1_acausal[k] = constant[2]*constant[4]/(float(1.00)-constant[4])*f[k];
                        }
                    }
    
                    for(k=0;k<channels;k++){
                        delta = delta + float(1.00)*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
                    }
    
                    delta = delta*s_quotient +1;
                    delta = sqrt(delta);
    
                    b_delta.real(cosf(delta*constant[9].real()));
                    b_delta.imag(sinf(delta*constant[9].real()));
                    b_delta = b_delta*powf(constant[11].real(),delta);
                 
                    
    
                    aux = (b_delta-float(1.00))/(constant[5]*delta);
                    for(k = 0;k <channels; k++){
                        g0[k] = b_delta*constant[1]*f[k]+ b_delta*prevg0_acausal[k];
                        g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
                    }
    
                    b_delta.real(cosf(delta*constant[10].real()));
                    b_delta.imag(sinf(delta*constant[10].real()));
                    b_delta = b_delta*powf(constant[12].real(),delta);
    
                    
    
                    aux= (b_delta-float(1.00))/(constant[7]*delta);
                    for(k = 0;k <channels; k++){
                        g1[k] = b_delta*constant[2]*f[k]+ b_delta*prevg1_acausal[k];
                        g1[k] = g1[k] + (aux - constant[8]*constant[4])*f[k] - (aux - constant[8]*b_delta)*f_prev[k];
                    }
        
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                        prevg0_acausal[k] = g0[k];
                        prevg1_acausal[k] = g1[k];
                    }
    
                }
            
            }


            for(i = i_end-1; i>= i_start; i=i-1){
                delta=0;
                buffer = simg[block_i*(window_w+ sharedpad) + threadIdx.x];

                f[0] = buffer.x;
                f[1] = buffer.y;
                f[2] = buffer.z;

                if(i==height-1){
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                        prevg0_acausal[k] = constant[1]*constant[3]/(float(1.00)-constant[3])*f[k];
                        prevg1_acausal[k] = constant[2]*constant[4]/(float(1.00)-constant[4])*f[k];
                    }
                }

                for(k=0;k<channels;k++){
                    delta = delta + float(1.00)*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
                }

                delta = delta*s_quotient +1;
                delta = sqrt(delta);

                b_delta.real(cosf(delta*constant[9].real()));
                b_delta.imag(sinf(delta*constant[9].real()));
                b_delta = b_delta*powf(constant[11].real(),delta);
             
                

                aux = (b_delta-float(1.00))/(constant[5]*delta);
                for(k = 0;k <channels; k++){
                    g0[k] = b_delta*constant[1]*f[k]+ b_delta*prevg0_acausal[k];
                    g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
                }

                b_delta.real(cosf(delta*constant[10].real()));
                b_delta.imag(sinf(delta*constant[10].real()));
                b_delta = b_delta*powf(constant[12].real(),delta);

                

                aux= (b_delta-float(1.00))/(constant[7]*delta);
                for(k = 0;k <channels; k++){
                    g1[k] = b_delta*constant[2]*f[k]+ b_delta*prevg1_acausal[k];
                    g1[k] = g1[k] + (aux - constant[8]*constant[4])*f[k] - (aux - constant[8]*b_delta)*f_prev[k];
                }
    
                for(k=0;k<channels;k++){
                    f_prev[k] = f[k];
                    prevg0_acausal[k] = g0[k];
                    prevg1_acausal[k] = g1[k];
                }
    
                aux.real(int(abs((g0[0]+g1[0]).real())));
                if(aux.real()>255){
                    buffer.x = 255;
                }
                else{ buffer.x = aux.real();}

                aux.real(int(abs((g0[1]+g1[1]).real())));
                if(aux.real()>255){
                    buffer.y = 255;
                }
                else{ buffer.y = aux.real();}

                aux.real(int(abs((g0[2]+g1[2]).real())));
                if(aux.real()>255){
                    buffer.z = 255;
                }
                else{ buffer.z = aux.real();}

                simg[block_i*(window_w+ sharedpad) + threadIdx.x] = buffer;
              
                block_i=block_i-1;
            }
            
        }




        __syncthreads();
        if(threadIdx.x<j_end-j_start)
        write_shareint_to_block_v(simg, outputimage,i_start,i_end,  j_start+threadIdx.x, threadIdx.x, width,  channels, window_w, sharedpad);
        __syncthreads();



        i_end = i_start;
        i_start = i_start - window_h;
        if(i_start<0){
            i_start = 0;
        }
    }
}

