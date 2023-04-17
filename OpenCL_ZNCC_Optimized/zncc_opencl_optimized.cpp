#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <lodepng.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <Windows.h>

// cl_info struct type to hold reused opencl objects
struct cl_info {
    cl::Context context;   
    cl::Program program;
    cl::Device device;
    cl::CommandQueue queue;
    cl::Event profEvent;
} cl_info_obj;

cl::Image2D EnqueueGrayScaleConversion(std::vector<unsigned char>& image, unsigned int width, unsigned int height)
{
    // setup two formats for the Image2D objects
    cl::ImageFormat rgbaFormat{ CL_RGBA, CL_UNSIGNED_INT8 };
    cl::ImageFormat grayscaleFormat{ CL_DEPTH, CL_UNSIGNED_INT8 };
    // create input image object, which is read_only and has a format of RGBA + 8 bit depth
    cl::Image2D inputImage(cl_info_obj.context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_USE_HOST_PTR, rgbaFormat, width, height, 0, image.data());
    // create output image object, which is read and write so that it can be reused by next kernel; use format of grayscale + 8 bit depth
    cl::Image2D outputImageGray(cl_info_obj.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, grayscaleFormat, width, height);
    // create a kernel object from the grayscale kernel
    cl::Kernel kernelGrayscale(cl_info_obj.program, "convert_grayscale");

    // set arguments
    kernelGrayscale.setArg(0, inputImage);
    kernelGrayscale.setArg(1, outputImageGray);

    // queue the grayscale kernel
    cl_info_obj.queue.enqueueNDRangeKernel(kernelGrayscale, cl::NullRange, cl::NDRange(width, height), cl::NullRange, 0, &cl_info_obj.profEvent);
    cl_info_obj.profEvent.wait();

    // print profiling
    double runTime = (double)(cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "Grayscale conversion execution time in microseconds " << runTime / (float)10e3 << std::endl;

    return outputImageGray;
}

cl::Buffer EnqueueResizeImage(cl::Image2D image, unsigned int width, unsigned int height, unsigned int resizeFactor)
{
    // create output buffer object for image so that it can be accessed easier in the next part of the pipeline
    // read_write access given, so that buffer can be reused as input
    cl::Buffer outputImageBuffResized(cl_info_obj.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(unsigned char) * (width * height));
    cl::Kernel kernelResize(cl_info_obj.program, "resize_image");

    // set arguments
    kernelResize.setArg(0, resizeFactor);
    kernelResize.setArg(1, image);
    kernelResize.setArg(2, outputImageBuffResized);

    // queue the resizing kernel
    cl_info_obj.queue.enqueueNDRangeKernel(kernelResize, cl::NullRange, cl::NDRange(width, height), cl::NullRange, 0, &cl_info_obj.profEvent);
    cl_info_obj.profEvent.wait();

    // print profiling
    double runTime = (double)(cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "Resize execution time in microseconds " << runTime / (float)10e3 << std::endl;

    return outputImageBuffResized;
}

cl::Buffer EnqueueZNCC(const cl::Buffer leftImage,
    const cl::Buffer rightImage,
    int width, int height,
    int windowSize, int maxDisparity,
    char isLeftImage = 1)
{
    // create buffer with read/write access so that it can be reused
    cl::Buffer disparityMap(cl_info_obj.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(unsigned int) * (width * height));
    cl::Kernel kernelZNCC(cl_info_obj.program, "calc_zncc");

    int halfWindowSize = (windowSize - 1) / 2;

    // set arguments
    kernelZNCC.setArg(0, halfWindowSize);
    kernelZNCC.setArg(1, isLeftImage);
    kernelZNCC.setArg(2, leftImage);
    kernelZNCC.setArg(3, rightImage);
    kernelZNCC.setArg(4, disparityMap);

    // queue the zncc kernel
    cl_info_obj.queue.enqueueNDRangeKernel(kernelZNCC, cl::NullRange, cl::NDRange(width, height, maxDisparity), cl::NDRange(maxDisparity), 0, &cl_info_obj.profEvent);
    cl_info_obj.profEvent.wait();

    // print profiling
    double runTime = (double)(cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "ZNCC execution time in microseconds " << runTime / (float)10e3 << std::endl;

    return disparityMap;
}

cl::Buffer EnqueueCrossCheck(const cl::Buffer dispMapLeft,
    const cl::Buffer dispMapRight,
    const int width, const int height, 
    const int crossDiff)
{
    // create buffer with read/write access so that it can be reused
    cl::Buffer crossCheckedImage(cl_info_obj.context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(unsigned int) * (width * height));
    cl::Kernel kernelCrossCheck(cl_info_obj.program, "cross_check");

    // set arguments
    kernelCrossCheck.setArg(0, crossDiff);
    kernelCrossCheck.setArg(1, dispMapLeft);
    kernelCrossCheck.setArg(2, dispMapRight);
    kernelCrossCheck.setArg(3, crossCheckedImage);

    // queue the cross check kernel
    cl_info_obj.queue.enqueueNDRangeKernel(kernelCrossCheck, cl::NullRange, cl::NDRange(width, height), cl::NullRange, 0, &cl_info_obj.profEvent);
    cl_info_obj.profEvent.wait();

    // print profiling
    double runTime = (double)(cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "Cross-checking execution time in microseconds " << runTime / (float)10e3 << std::endl;

    return crossCheckedImage;
}

cl::Buffer EnqueueOcclusionFilling(const cl::Buffer crossCheckedImage, 
    const int width, const int height, const int nCount, const int workGroupSize)
{
    // no buffer created here, as the input image's invalid pixels are filled in directly
    cl::Kernel kernelFilling(cl_info_obj.program, "occlusion_filling");

    if (nCount * nCount > workGroupSize)
    {
        std::cout << "Warning: cross-check window size (nCount * nCount) bigger than allowed work group size." << std::endl;
        std::cout << "Possible undefined behavior.Choose smaller nCount." << std::endl;
    }

    // set arguments
    kernelFilling.setArg(0, nCount);
    kernelFilling.setArg(1, workGroupSize * sizeof(int), NULL);
    kernelFilling.setArg(2, crossCheckedImage);

    // queue the occlusion filling kernel
    cl_info_obj.queue.enqueueNDRangeKernel(kernelFilling, cl::NullRange, cl::NDRange(width, height), cl::NullRange, 0, &cl_info_obj.profEvent);
    cl_info_obj.profEvent.wait();

    // print profiling
    double runTime = (double)(cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "Occlusion filling execution time in microseconds " << runTime / (float)10e3 << std::endl;

    return crossCheckedImage;
}

cl::Buffer EnqueueNormalizeToChar(const cl::Buffer filledImage, 
    const int width, const int height, const int ndisp)
{
    // buffer with write only permission as it will not be reused in the future anymore
    cl::Buffer normImage(cl_info_obj.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned char) * (width * height));
    cl::Kernel kernelNorm(cl_info_obj.program, "normalize_to_char");

    // set arguments
    kernelNorm.setArg(0, ndisp);
    kernelNorm.setArg(1, filledImage);
    kernelNorm.setArg(2, normImage);

    // queue the resizing kernel
    cl_info_obj.queue.enqueueNDRangeKernel(kernelNorm, cl::NullRange, width * height, cl::NullRange, 0, &cl_info_obj.profEvent);
    cl_info_obj.profEvent.wait();

    // print profiling
    double runTime = (double)(cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cl_info_obj.profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "Image normalization execution time in microseconds " << runTime / (float)10e3 << std::endl;

    return normImage;
}

int main()
{
    // from calib.txt - downsized
    // each pixel in the downsampled image corresponds to a larger area in the original image
    // reducing the resolution of the images reduces the maximum disparity that can be reliably estimated
    // Disparity value represents the number of pixels that one image point is shifted relative to the other image point,
    // so the maximum disparity value is directly related to the image resolution.
    // To account for this reduction in resolution, we need to adjust the maximum disparity value by the same factor that we used to downsample the image

    int ndisp = 260;
    unsigned int resizeFactor = 4;
    int winSize = 11;
    int neighbours = 32;
    int crossDiff = 32;

    // setup inputs and outputs
    const char* leftImgName = "../img/im0.png";
    const char* rightImgName = "../img/im1.png";

    const char* depthmapOut = "../img/cl_depthmap_optimized.png";

    // create containers for raw images
    std::vector<unsigned char> leftImage;
    std::vector<unsigned char> rightImage;
    unsigned int width, height;

    // decode images
    unsigned int error = lodepng::decode(leftImage, width, height, leftImgName, LCT_RGBA, 8);
    if (error) std::cout << "decoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

    error = lodepng::decode(rightImage, width, height, rightImgName, LCT_RGBA, 8);
    if (error) std::cout << "decoder error second image: " << error << ": " << lodepng_error_text(error) << std::endl;

    try
    {
        // create the program
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        auto platform = platforms.front();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        auto device = devices.front();

        std::cout << "------------HARDWARE INFORMATION------------" << std::endl;
        auto platName = platform.getInfo<CL_PLATFORM_NAME>();
        auto devVersion = device.getInfo<CL_DEVICE_VERSION>();
        auto devDriver = device.getInfo<CL_DRIVER_VERSION>();
        auto devCVersion = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();

        std::cout << "Platform count: " << platforms.size() << std::endl;
        std::cout << "Platform 1 name: " << platName << std::endl;
        std::cout << "Device count on platform 1: " << devices.size() << std::endl;
        std::cout << "Hardware version: " << devVersion << std::endl;
        std::cout << "Driver version: " << devDriver << std::endl;
        std::cout << "OpenCL C version: " << devCVersion << std::endl;

        std::cout << "------------DEVICE INFORMATION------------" << std::endl;
        auto devInfo = device.getInfo<CL_DEVICE_NAME>();
        auto devMemType = device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();
        auto devMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        auto devPCunits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        auto devClockFreq = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
        auto devConstBuffSize = device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
        auto devWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        auto devWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        auto devWorkItemDim = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
        auto devMaxReadImageArgs = device.getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>();

        std::cout << "Device information: " << devInfo << std::endl;
        std::cout << "Local memory types: " << devMemType << std::endl;
        std::cout << "Local memory size: " << devMemSize << std::endl;
        std::cout << "Parallel Compute units: " << devPCunits << std::endl;
        std::cout << "Max clock frequency: " << devClockFreq << std::endl;
        std::cout << "Max constant buffer size: " << devConstBuffSize << std::endl;
        std::cout << "Work group size: " << devWorkGroupSize << std::endl;
        for (size_t i = 0; i < devWorkItemSizes.size(); i++)
        {
            std::cout << "Work item " << i << " size: " << devWorkItemSizes[i] << std::endl;
        }
        std::cout << "Max Work Item Dimensions: " << devWorkItemDim << std::endl;
        std::cout << "Max read image arguments: " << devMaxReadImageArgs << std::endl;

        std::cout << "------------IMPLEMENTATION------------" << std::endl;

        // start timing execution time
        LARGE_INTEGER start, end, frequency;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&start);

        // Create source
        // Note: All kernels are stored in one kernel file as
        // OpenCL C++ seems to have an issue if you pass a second kernel file
        std::ifstream kernelFile("../kernels/zncc_kernels_optimized.cl");
        std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

        cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

        // create context and build program
        cl::Context context(devices);
        cl::Program program(context, sources);

        program.build("-cl-std=CL2.0"); //#TODO: note down

        // create command queue with profiling enabled
        cl_command_queue_properties properties[]{ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        cl::CommandQueue queue(context, device, properties);
        cl::Event profEvent;

        // fill in custom struct
        cl_info_obj.context = context;
        cl_info_obj.program = program;
        cl_info_obj.device = device;
        cl_info_obj.queue = queue;
        cl_info_obj.profEvent = profEvent;

        // Kernel logic
        //// Grayscale conversion
        std::cout << "Converting left image to grayscale..." << std::endl;
        auto outputImageGrayLeft = EnqueueGrayScaleConversion(leftImage, width, height);
        std::cout << "Converting right image to grayscale..." << std::endl;
        auto outputImageGrayRight = EnqueueGrayScaleConversion(rightImage, width, height);

        //// Rescaling
        // update values depending on resolution
        int oldWidth = width;
        width = width / resizeFactor;
        height = height / resizeFactor;
        ndisp = ndisp * (static_cast<float>(width) / oldWidth);

        // enqueue resizing
        std::cout << "Resizing left image to 1/16 size..." << std::endl;
        auto outputImageResizedLeft = EnqueueResizeImage(outputImageGrayLeft, width, height, resizeFactor);
        std::cout << "Resizing right image to 1/16 size..." << std::endl;
        auto outputImageResizedRight = EnqueueResizeImage(outputImageGrayRight, width, height, resizeFactor);
        
        // enqueue ZNCC
        std::cout << "Applying ZNCC to left image..." << std::endl;
        auto outputZNCCLeft = EnqueueZNCC(outputImageResizedLeft, outputImageResizedRight, width, height, winSize, ndisp);
        std::cout << "Applying ZNCC to right image..." << std::endl;
        auto outputZNCCRight = EnqueueZNCC(outputImageResizedRight, outputImageResizedLeft, width, height, winSize, ndisp, -1);
        
        // enqueue cross-check
        std::cout << "Applying cross-check..." << std::endl;
        auto outputCrossCheck = EnqueueCrossCheck(outputZNCCLeft, outputZNCCRight, width, height, crossDiff);

        // enqueue occlusion filling
        std::cout << "Applying occlusion filling..." << std::endl;
        auto outputOcclusionFilling = EnqueueOcclusionFilling(outputCrossCheck, width, height, neighbours, devWorkGroupSize);

        //// enqueue normalization
        std::cout << "Applying image normalization..." << std::endl;
        auto outputNorm = EnqueueNormalizeToChar(outputOcclusionFilling, width, height, ndisp);

        // read the normalized depthmap output and put it into a vector
        cl::Event readEvent;
        std::vector<unsigned char> normImage(width * height);
        cl_info_obj.queue.enqueueReadBuffer(outputNorm, CL_TRUE, 0, sizeof(unsigned char) * normImage.size(), normImage.data(), 0, &readEvent);

        // print profiling
        double transferTime = (double)(readEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - readEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        std::cout << "Final read bus transfer time in microseconds " << transferTime / (float)10e3 << std::endl;

        // end execution timing and print
        QueryPerformanceCounter(&end);
        double elapsed_time;
        elapsed_time = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
        std::cout << "Total elapsed time: " << elapsed_time * 1000000 << " microseconds\n";

        error = lodepng::encode(depthmapOut, normImage, width, height, LCT_GREY, 8);
        if (error) std::cout << "encoder error: " << error << ": " << lodepng_error_text(error) << std::endl;

    }
    catch (cl::Error err) {
        std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err.err()
            << ")"
            << std::endl;
    }

}