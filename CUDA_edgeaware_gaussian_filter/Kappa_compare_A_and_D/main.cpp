#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../stbi_headers/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stbi_headers/stb_image_write.h"
#include "../stbi_headers/stb_image_resize.h"


float  *load_matrixA(int size){
    FILE *arq;
    float *matrix = (float*)malloc(sizeof(float)*size);
    arq = fopen("../outputmatrix/A.txt","r");
    int i;
    for(i=0;i<size;i++){
        fscanf(arq,"%f\n",&matrix[i]);
    }
    fclose(arq);
    return matrix;
}

void  load_matrixD(float *matrix,int size){
    FILE *arq;
    arq = fopen("../outputmatrix/D.txt","r");
    int i;

    for(i=0;i<size;i++){
        fscanf(arq,"%f\n",&matrix[i]);
    }

    fclose(arq);

}

void concat_sigmas(char *execpath,char *sigma, char *sigma_r){
    strcat(execpath," "); strcat(execpath,sigma);
    strcat(execpath," "); strcat(execpath,sigma_r);
}

int main(int argc, char *argv[]){
    float step,kappa,begin,end;
    int status,i,height,width,channels;
    char *execpath;
    char buf[6];
    unsigned char *o_img;
    FILE *arq2;
    float *img_vectorD,*img_vectorA;
    double MSE;
    double PSNR;
    int k;
    float sigma, sigma_r;

    char imagefile[20];
    char filelocation[100] = "../images/";

    if(argc<4){
        printf("error, not enough arguments\n");
        return -1;
    }

    if(argc == 6){
        sigma = atof(argv[4]);
        sigma_r = atof(argv[5]);
    }

    begin = atof(argv[1]);
    end = atof(argv[2]);
    step = atof(argv[3]);

    //load original image
    FILE *arq;
    arq = fopen("../parameters.txt","r");
    if((fscanf(arq,"image %s\n",imagefile)!=1)){
        printf("error while reading parameters\n");
        return -1;
    }
    fclose(arq);
    strcat(filelocation,imagefile);

    o_img = stbi_load(filelocation,&width,&height,&channels,0); 
    execpath = (char*)malloc(100*sizeof(char));
    //execute algorithmA to obtain reference image
    if(argc<5){
    status = system("./../Algorithm_A/main 1");
    }
    else{
        //get cmd string
        for(i=0;i<100;i++)
        execpath[i]='\0';
        strcat(execpath,"./../Algorithm_A/main 1");
        concat_sigmas(execpath,argv[4],argv[5]);
        printf("A exec = %s \n",execpath);
        status = system(execpath);
        //
    }

    img_vectorA = load_matrixA(width*height*channels);

    arq = fopen("../Kappa_compare_A_and_D/MSE.tsv","w");

    fprintf(arq,"k	MSE(%.0f,%.0f)\n",sigma,sigma_r);

    img_vectorD = (float*)malloc(width*height*channels*sizeof(float));

    //execute algorithmD with different kappa values
    for(kappa = begin; kappa <= end; kappa+=step){
        for(i=0;i<100;i++)
        execpath[i]='\0';

        strcat(execpath,"./../Algorithm_D/main ");
        gcvt(kappa,6,buf);
        strcat(execpath,buf);

        if(argc==6)
        concat_sigmas(execpath,argv[4],argv[5]);
    
        printf("path = %s\n",execpath);
        //execute algorithmC with kappa
        status = system(execpath);
        //read images and compare
        load_matrixD(img_vectorD,width*height*channels);  



        MSE=0;
        //compute average squared error
        for(i=0;i<height*width*channels;i++){
            MSE = MSE + (img_vectorA[i]-img_vectorD[i])*(img_vectorA[i]-img_vectorD[i]);
        }

        MSE = MSE/(width*height*channels*1.0);
        PSNR = 20*log10f(255.0/MSE);

        
        printf("MSE = %f , PSNR = %f\n",MSE,PSNR);
        fprintf(arq,"%f\t%f\n",kappa,MSE);



        free(execpath);
    }
    fclose(arq);

    //status = system("gnuplot generateimage.gnu  > mse_plot.pdf");

    free(img_vectorA);
    return 0;
}
