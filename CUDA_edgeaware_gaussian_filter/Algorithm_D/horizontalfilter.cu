#include <iostream>
#include <thrust/complex.h>
#include "header.cuh"
#include <vector>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

using namespace std;


__device__ void read_blockint_to_shareint(uchar4 *simg, uchar4 *img, int i_start, int i_end, int col_index, int block_col_index, int width, int channels, int window_w, int pad){
    int i;
    for(i= threadIdx.x/32;i<i_end-i_start;i=i+4){
        simg[i*(window_w+pad) + block_col_index]=img[(i_start+i)*(width) + col_index];
    }
}

__device__ void write_shareint_to_block( uchar4 *simg, uchar4 *img, int i_start, int i_end, int col_index, int block_col_index, int width, int channels, int window_w,int pad){
    int i;

    for(i=threadIdx.x/32;i<i_end-i_start;i=i+4){
        img[(i_start+i)*(width) + col_index]=simg[i*(window_w+pad) + block_col_index];
    }
}

__global__ 
void gaussian_filter_kernel_horizontal_anticausal(uchar4  *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4  *img,int width, int height,int channels, int window_w, int window_h, int sharedpad, float kappa){

    int i,j,k,j_start=0,j_end = window_w,step,i_start,i_end;
    uchar4 buffer;
    int block_j;
    float f[3],f_prev[3],delta;
    float dist;
    extern __shared__  uchar4 simg[]; 
    thrust::complex<float> prevg0_acausal[3],prevg1_acausal[3],g0[3],g1[3];
    thrust::complex<float> b_delta,aux;

    i = threadIdx.x + window_h*blockIdx.x;
    
    j_end = width;
    j_start = j_end - window_w;
    while(j_end > 0){ 
        aux.real(0);
        aux.imag(0);
        

        i_start = window_h*blockIdx.x;
        i_end = i_start + window_h;
        if(i_end> height){
            i_end = height;
        }
        
        if(threadIdx.x%32<j_end-j_start)
        read_blockint_to_shareint(simg, img,i_start,i_end,  j_start+threadIdx.x%32, threadIdx.x%32, width,  channels, window_w, sharedpad);
        __syncthreads();


        if(i<height){ 
            block_j = (j_end-j_start)-1;
            
            if(kappa>=0){

                //GET APPROXIMATION FOR INITIAL CONDITIONS
                    //compute extended length
                    dist=0;
                    j=j_end-1;

                    buffer = img[i*width+j_end-1];
                    f_prev[0] = buffer.x;
                    f_prev[1] = buffer.y;
                    f_prev[2] = buffer.z;


                    while(j<width-1 && (dist<sigma_h*kappa)){
                        j=j+1;
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
                        if(j==0){
                            delta=1;
                        }
                        dist = dist + delta;
                        for(k=0;k<channels;k++){
                            f_prev[k] = f[k];
                        }
                    }
                    // COMPUTE INITIAL CONDITIONS 
                    dist=j;

                    for(j = j; j>= j_end; j= j-1){
                
                        delta=0;
                        buffer = img[i*width +j];
                        f[0] = buffer.x;
                        f[1] = buffer.y;
                        f[2] = buffer.z;
        
                        if(j == width-1){
                            for(k=0;k<channels;k++){
                                f_prev[k] = f[k];
                                prevg0_acausal[k] = constant[1]*constant[3]/(float(1.00)-constant[3])*f[k];
                                prevg1_acausal[k] = constant[2]*constant[4]/(float(1.00)-constant[4])*f[k];
                            }
                        }
                        for(k=0;k<channels;k++){
                            delta = delta + float(1.00)*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
                        }
        
                        delta = delta*s_quotient +float(1.00);
                        delta = sqrt(delta);
        
            
                        //b_delta = pow(constant[3],delta);
                        b_delta.real(__cosf(delta*constant[9].real()));
                        b_delta.imag(__sinf(delta*constant[9].real()));
                        b_delta = b_delta*__powf(constant[11].real(),delta);
        
        
                        aux = (b_delta-float(1.00))/(constant[5]*delta);
                        
            
                        for(k = 0;k <channels; k++){
                            g0[k] = constant[1]*b_delta*f_prev[k]+ b_delta*prevg0_acausal[k];
                            g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
         
                        }
                        //b_delta = pow(constant[4],delta);
                        b_delta.real(__cosf(delta*constant[10].real()));
                        b_delta.imag(__sinf(delta*constant[10].real()));
                        b_delta = b_delta*__powf(constant[12].real(),delta);
        
                        aux = (b_delta-float(1.00))/(constant[7]*delta);
                       
                        for(k = 0;k <channels; k++){
        
                            g1[k] = constant[2]*b_delta*f_prev[k]+ b_delta*prevg1_acausal[k];
                            
                            g1[k] = g1[k] + (aux - constant[8]*constant[4])*f[k] - (aux - constant[8]*b_delta)*f_prev[k];
                        }
            
                        for(k=0;k<channels;k++){
                            f_prev[k] = f[k];
                            prevg0_acausal[k] = g0[k];
                            prevg1_acausal[k] = g1[k];
                        }
        
                       
                    }
            
                }
    
    


            for(j = j_end-1; j>= j_start; j= j-1){
                
                delta=0;
                buffer = simg[threadIdx.x*(window_w + sharedpad) + block_j];
                f[0] = buffer.x;
                f[1] = buffer.y;
                f[2] = buffer.z;

                if(j == width-1){
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                        prevg0_acausal[k] = constant[1]*constant[3]/(float(1.00)-constant[3])*f[k];
                        prevg1_acausal[k] = constant[2]*constant[4]/(float(1.00)-constant[4])*f[k];
                    }
                }
                for(k=0;k<channels;k++){
                    delta = delta + float(1.00)*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
                }

                delta = delta*s_quotient +float(1.00);
                delta = sqrt(delta);

                b_delta.real(__cosf(delta*constant[9].real()));
                b_delta.imag(__sinf(delta*constant[9].real()));
                b_delta = b_delta*__powf(constant[11].real(),delta);


                aux = (b_delta-float(1.00))/(constant[5]*delta);
                
    
                for(k = 0;k <channels; k++){
                    g0[k] = constant[1]*b_delta*f_prev[k]+ b_delta*prevg0_acausal[k];
                    g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
 
                }
                b_delta.real(__cosf(delta*constant[10].real()));
                b_delta.imag(__sinf(delta*constant[10].real()));
                b_delta = b_delta*__powf(constant[12].real(),delta);

                aux = (b_delta-float(1.00))/(constant[7]*delta);
               
                for(k = 0;k <channels; k++){

                    g1[k] = constant[2]*b_delta*f_prev[k]+ b_delta*prevg1_acausal[k];
                    
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

                simg[threadIdx.x*(window_w + sharedpad) + block_j] = buffer;
 
                block_j = block_j -1;
            }
    
    
        }


        __syncthreads();
        if(threadIdx.x%32<j_end-j_start)
        write_shareint_to_block(simg, outputimage,i_start,i_end,  j_start+threadIdx.x%32, threadIdx.x%32, width,  channels, window_w, sharedpad);
        __syncthreads();

        j_end = j_start;
        j_start = j_start - window_w;
        if(j_start<0){
            j_start = 0;
        }
        

    }
}


__global__ 
void gaussian_filter_kernel_horizontal_causal(uchar4  *outputimage,float sigma_h,float s_quotient,thrust:: complex <float> *constant, uchar4  *img,int width, int height,int channels, int window_w, int window_h, int sharedpad, float kappa){

    int i,j,k,j_start=0,j_end = window_w,i_start,i_end;
    float dist;
    uchar4 buffer;
    int block_j;
    float f[3],f_prev[3],delta;
    extern __shared__  uchar4 simg[]; 
    thrust::complex<float> prevg0_causal[3],prevg0_acausal[3],prevg1_acausal[3],prevg1_causal[3],g0[3],g1[3];
    thrust::complex<float> b_delta,aux;

    i = threadIdx.x + window_h*blockIdx.x;
    
    while(j_start < width){ 
        aux.real(0);
        aux.imag(0);
        j_end = j_start +window_w;
        
        if(j_end> width){
            j_end = width;
        }
        

        i_start = window_h*blockIdx.x;
        i_end = i_start + window_h;
        if(i_end> height){
            i_end = height;
        }
        
        if(threadIdx.x%32<j_end-j_start)
        read_blockint_to_shareint(simg, img,i_start,i_end,  j_start+threadIdx.x%32, threadIdx.x%32, width,  channels, window_w, sharedpad);
        __syncthreads();


        if(i<height){             

            block_j = 0;
            if(kappa>=0){
            //GET APPROXIMATION FOR INITIAL CONDITIONS
                //compute extended length
                dist=0;
                j=j_start;
                while(j>0 && (dist<sigma_h*kappa)){
                    j=j-1;
                    delta=0;

                    buffer = img[i*width+j_start];
                    f_prev[0] = buffer.x;
                    f_prev[1] = buffer.y;
                    f_prev[2] = buffer.z;

                    buffer = img[i*width +j];
                    f[0] = buffer.x;
                    f[1] = buffer.y;
                    f[2] = buffer.z;
                    for(k=0;k<channels;k++){
                        delta = delta + 1.00*((f[k]-f_prev[k])*(f[k]-f_prev[k]));
        
                    }
                    delta = delta*s_quotient +1;
                    delta = sqrt(delta);
                    if(j==0){
                        delta=1;
                    }

                    dist = dist + delta;
                    for(k=0;k<channels;k++){
                        f_prev[k] = f[k];
                    }
                }
                // COMPUTE INITIAL CONDITIONS 
                dist = j;
                for(j=j;j<= j_start;j++){

                    delta=0;
                        buffer = img[i*width +j];
                        f[0] = buffer.x;
                        f[1] = buffer.y;
                        f[2] = buffer.z;

                    if(j==dist){
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
    
                    b_delta.real(__cosf(delta*constant[9].real()));
                    b_delta.imag(__sinf(delta*constant[9].real()));
                    b_delta = b_delta*__powf(constant[11].real(),delta);
                 
                    
    
                    aux = (b_delta-float(1.00))/(constant[5]*delta);
                    for(k = 0;k <channels; k++){
                        g0[k] = constant[1]*f[k]+ b_delta*prevg0_causal[k];
                        g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
                    }
    
                    b_delta.real(__cosf(delta*constant[10].real()));
                    b_delta.imag(__sinf(delta*constant[10].real()));
                    b_delta = b_delta*__powf(constant[12].real(),delta);
    
                    
    
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


            for(j = j_start; j< j_end; j++){
                delta=0;
                    buffer = simg[threadIdx.x*(window_w + sharedpad)+block_j];
                    f[0] = buffer.x;
                    f[1] = buffer.y;
                    f[2] = buffer.z;

                if(j==0){
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

                b_delta.real(__cosf(delta*constant[9].real()));
                b_delta.imag(__sinf(delta*constant[9].real()));
                b_delta = b_delta*__powf(constant[11].real(),delta);
             
                

                aux = (b_delta-float(1.00))/(constant[5]*delta);
                for(k = 0;k <channels; k++){
                    g0[k] = constant[1]*f[k]+ b_delta*prevg0_causal[k];
                    g0[k] = g0[k] + (aux - constant[6]*constant[3])*f[k] - (aux - constant[6]*b_delta)*f_prev[k];
                }

                b_delta.real(__cosf(delta*constant[10].real()));
                b_delta.imag(__sinf(delta*constant[10].real()));
                b_delta = b_delta*__powf(constant[12].real(),delta);

                

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

                simg[threadIdx.x*(window_w + sharedpad) + block_j] = buffer;

                block_j++;
            }
            
        }



        __syncthreads();
        if(threadIdx.x%32<j_end-j_start)
        write_shareint_to_block(simg, outputimage,i_start,i_end,  j_start+threadIdx.x%32, threadIdx.x%32, width,  channels, window_w, sharedpad);
        __syncthreads();

        j_start = j_end;

    }
}
