#include <iostream>
#include <thrust/complex.h>
using namespace std;
#define STB_IMAGE_IMPLEMENTATION
#include "../stbi_headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stbi_headers/stb_image_write.h"
#include "../stbi_headers/stb_image_resize.h"
#include <time.h>
#include <cuda_runtime.h>
#include "header.cuh"
#include <cstring>
#include <chrono> 
using namespace std::chrono; 


int main(int argc, char* argv[]){

    int width,height,channels;
    float sigma = 70.00;
    float sigma_r;
    float kappa = 2.0;
    FILE *arq;
    int line_count;
    char imagefile[20];
    char filelocation[100] = "../images/";
    //cuda window parameters
    int window_h = 128, window_w = 24;

    //Read parameters
    arq = fopen("../parameters.txt","r");
    if((fscanf(arq,"image %s\n",imagefile)!=1) ||(fscanf(arq,"sigma %f\n",&sigma)!=1) || 
    (fscanf(arq,"sigmar %f\n",&sigma_r) !=1)||(fscanf(arq,"kappa %f\n",&kappa)!=1)|| (fscanf(arq,"blocks_per_line %i\n",&line_count) !=1)){
        printf("error while reading parameters\n");
        return -1;
    }
    fclose(arq);
    strcat(filelocation,imagefile);

    unsigned char *img_vector = stbi_load(filelocation,&width,&height,&channels,0);     //CARREGA JPG USANDO STBI
    uchar4 *i_image;

    if(img_vector==NULL){ //could not read image
	printf("erro\n");
        return -1;
    }
    
    unsigned char *outputimage = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));

    i_image = convert_uimg_to_uchar4( width,  height,  channels, img_vector);
    
    free(img_vector);

    image_filter2d(sigma,sigma_r , i_image, width, height,  channels, window_w,  window_h, kappa, line_count);

    //printf("loaded image with w = %i h = %i and c = % i channels \n",width,height,channels);
    
    
    
    transfer_uchar4_uint(width,height,channels,i_image,outputimage);

    stbi_write_jpg("../output/resultC.jpg",width,height,channels,&outputimage[0],100);
    
    free(outputimage);
    cudaFree(i_image);
    return 0;

}

