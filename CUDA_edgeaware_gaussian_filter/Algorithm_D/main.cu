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
    float kappa;
    float sigma;
    float sigma_r;
    int line_count; //unused
    float exectime;
    int window_h = 128, window_w = 24;
    FILE *arq;
    char imagefile[20];
    char filelocation[100] = "../images/";

    arq = fopen("../parameters.txt","r");
    if((fscanf(arq,"image %s\n",imagefile)!=1) ||(fscanf(arq,"sigma %f\n",&sigma)!=1) || 
    (fscanf(arq,"sigmar %f\n",&sigma_r) !=1)||(fscanf(arq,"kappa %f\n",&kappa)!=1)|| (fscanf(arq,"blocks_per_line %i\n",&line_count) !=1)){
        printf("error while reading parameters\n");
        return -1;
    }

    strcat(filelocation,imagefile);

    unsigned char *img_vector = stbi_load(filelocation,&width,&height,&channels,0);     
    uchar4 *i_image;

    if(img_vector==NULL){ 
	printf("Error: no image loaded\n");
        return -1;
    }
    
    //Set kappa to argv value if not NULL
    if(argc>1)
    kappa = atof(argv[1]);
    
    //set sigma_r and sigma values if values chosen on argv
    if(argc>2)
    sigma = atof(argv[2]);
    if(argc>3)
    sigma_r = atof(argv[3]);

    unsigned char *outputimage = (unsigned char*)malloc(width*height*channels*sizeof(unsigned char));

    i_image = convert_uimg_to_uchar4( width,  height,  channels, img_vector);
    
    free(img_vector);

    image_filter2d(&exectime,sigma_r,sigma,  i_image, width, height,  channels, window_w,  window_h,kappa);
    
    transfer_uchar4_uint(width,height,channels,i_image,outputimage);

    stbi_write_jpg("../output/resultD.jpg",width,height,channels,&outputimage[0],100);
    
    printf("exectime\t%f\n",exectime);

    //write ucharmatrix if comparing algorithms
    int i;
    if(argv[1]!=NULL){
        arq = fopen("../outputmatrix/D.txt","w");
        for(i=0;i<height*width;i++){
            fprintf(arq,"%i\n",i_image[i].x);
            fprintf(arq,"%i\n",i_image[i].y);
            fprintf(arq,"%i\n",i_image[i].z);
        }
        fclose(arq);
    }

    free(outputimage);
    cudaFree(i_image);
    return 0;

}
