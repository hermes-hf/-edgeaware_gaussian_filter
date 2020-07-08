# cuda_edgeaware_gaussian_filter
  Inside the folder are the 3 algorithms used throught the work. When given no argv inputs, each algorithm will run using the arguments given on the parameters.txt file, choosing an image inside the images folder acoording to the file name given.
  
  We provide a folder "images" with the images used in our work. Filtered images are placed in the output folder. Kappa_compare_A_and_D folder is used when comparing the MSE between the outputs.
  
  In the jupyter_folder we also provide with a jupyter notebook with the code used to generate the figures in the work.
  
  We make use of the stb image processing headers, provided in https://github.com/nothings/stb.
  
  Compatibility:
  - We used the nvcc compiler version 10.2.
  - For the jupyter notebook we used python 3.

  Input Parameters:
  - image: image file used.
  - sigma: size of the filter in pixels.
  - sigmar: size of the filter on RGB spectrum.
  - kappa: parameter used on algorithms C and D to determine approximation quality.
  - blocks_per_line: parameter to calibrate the amount of approximated regions.
  
  Finally, we provide with a makefile on the folder to compile every algorithm used.
