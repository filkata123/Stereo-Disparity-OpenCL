# Multiprocessor Programming Training Diary
**Name:** Filip Georgiev

**Student number:** 2208085

The purpose of this training diary is to showcase the student’s journey with OpenCL (in C++). The document will be split into the pre-set phases of the exercise project for the Multiprocessor Programming course, done in the University of Oulu.

This project was implemented with Visual Studio and consequently the MSVC compiler. An example [property list](GlobalSheet.props) for Visual Studio is given with necessary compiler flags. 
The property list also includes include and library directory paths, that need to be changed on every machine. Put your paths for OpenCL and Lodepng in the following two tags: **AdditionalIncludeDirectories** and **AdditionalLibraryDirectories**. (This has not been tested on a machine other than Windows 11)

The Nvidia CUDA OpenCL SDK and runtime were used for the implementations found in this repository.

## Start
Open MPP_Project.sln with Visual Studio, choose a project from the projects that will be listed in bold below, and set it as startup, before building and running.

The Lodepng library was imported as a submodule with the intention to decode and encode PNG images.

## Phase 0 (3 hours)
To get familiar with OpenCL, the [linked video](https://www.youtube.com/playlist?list=PLzy5q1NUJKCJocUKsRxZ0IPz29p38xeM-) tutorial series was watched. Watching the videos and following along, while also doing some online exploration took around 3 hours. 
## Phase 1 ()
### Step 1 (9 hours)
The **CPU_Moving_Filter** project contains the code for step 1 (lodepng read/write, grayscale and resize). 
The code also implements a 5x5 Gaussian blur moving filter on CPU. 

The original and processed images can be found in the ```img/``` folder. 
The two original images used are ```im{0/1}.png``` and were gotten from [here](https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Backpack-perfect/).

![](img/im0.png "Original image")

The original image, seen above, had RGBA colour type.
This meant that it had four channels for each colour and opacity, i.e. the size of the image would be 4 times bigger than its resolution (width * height).
This was taken into account while decoding with lodepng.
After converting the resulting raw vector to grayscale by adding up the R, G and B values and discarding the A value, its size would become the same as the resolution.

As per the assignment, the image was further downsized to 1/16th of its size by dividing the image into 4x4 blocks and using the average value of the block as the value of the new pixel.

![](img/im0_out.png "Resized grayscale image")

The processed image was then further manipulated by applying gaussian blur. This was done with the help of a 5x5 moving filter.
As can be seen below, the image is considerably blurrier.
![](img/im0_gauss_out.png "Gaussuan blur image")

The images are then decoded with lodepng. 
The results are displayed above.

Pure code can be found in [moving_filter.cpp](moving_filter.cpp).

This step took 9 hours, as the writer is not familiar with image processing.

### Step 2 (8.5 hours)
Step two is divided into two parts - implementing matrix addition with C on the CPU and then implementing the same with OpenCL on the GPU.
The C code can be found in the [Matrix_Addition_C](Matrix_Addition_C/) folder (together with compilation instructions).
It makes use of Windows.h's QueryPerformanceCounter functionality to track how long matrix addition takes.
The resulting time is shown below.

![](diary_img/matrix_addition_c.png "Execution time of C program")

The OpenCL implementation is in the **Matrix_Addition_OpenCL** project.
An issue encountered was trying to pass a 2D dynamic array into the buffer.
To resolve this issue, the array was flattend into a pointer to a 1D array.
The dimensions from the perspective of the kernel were still 2, but this is done so that the kernel has access to the rows and columns.
Implementing matrix addition with OpenCL made execution around 10 times faster, as seen below.

![](diary_img/matrix_addition_opencl.png "Execution time of OpenCL program")

The OpenCL code can be viewed in [matrix_addition.cpp](matrix_addition.cpp), while the respective kernel used in [kernels/add_matrix.cl](kernels/add_matrix.cl).

This step took around 8.5 hours, due to this being the first real attempt at OpenCL.

### Step 3 ()