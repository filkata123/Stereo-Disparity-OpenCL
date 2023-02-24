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
## Phase 1 (27 hours)
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

### Step 3 (5.5 hours)
Step three made use of all the knowledge obtained previously in this phase to create an OpenCL implementation, which converts an image to grayscale and applies a 5x5 moving filter, similarly to step 1.
However, no resizing was applied in this case.
Two kernels were created for this implementation. The first one, would take an iamge as an input and convert it to grayscale, while the second oneand would take the output of the previous kernel as an input, together with an gaussian filter matrix and apply the latter to the former, so as to achieve Gaussian blur.
It's important to note that the two kernels were kept in one kernel file, as C++ did not allow for them to be input separetely.

The OpenCl implementation is in the **OpenCL_Moving_Filter** project.
After decoding the raw image, similarly to step 1, it was passed to the first kernel.
As this implementation did not make use of a for loop, iterating every four indexes had to be done in a diffent way.
Each work item would check whether its ID is a multiple of four.
If it is, it would add up its own value and the next two ones, essentially adding up the R, G and B channels.
If not, it would just skip over to the next work item.

The second kernel did not need to change the moving filter code with the exclusion that it removed the iterations over the width and height of the image, similarly to the kernel from the previous step.

As the image was basically a 2D array it could be seen as a matrix from the previous step.
This meant that it had to be handled in a similar way.

The final result was encoded and can be found [here](img/imCV_out.png).
As the image is not resized, the blur is not as noticable, but can be seen if closely inspected.

The following information is printed on execution. 

![](diary_img/image_manipulation_opencl.png "Execution time and hw info of OpenCL program")

As can be seen, loading and saving the image takes much more time than the OpenCL-enabled processing functions.
Expectidely, the grayscale conversion is faster than the gaussian filter.
The bus transfer time, while reading the processed image, is also shown for both kernels.

The OpenCL code can be investigated in [image_manipulator.cpp](image_manipulator.cpp), while the kernels can be seen in [kernels/image_manipulator_kernels.cl](kernels/image_manipulator_kernels.cl).

This step took around 5.5 hours.

The whole phase took 27 hours, as an additional 4 hours were spent documenting the whole journey so far :)

## Phase 2 ()

3:30 hours for zncc algorithms check