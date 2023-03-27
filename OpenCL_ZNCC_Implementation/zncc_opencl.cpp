#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <lodepng.h>

#include <iostream>
#include <fstream>
#include <math.h> 
#include <vector>
#include <Windows.h>

cl::Image2D EnqueueGrayScaleConversion(cl::Context& context, cl::Program program, cl::Device device, std::vector<unsigned char>& image, unsigned int width, unsigned int height)
{
    cl::ImageFormat rgbaFormat{ CL_RGBA, CL_UNSIGNED_INT8 };
    cl::ImageFormat grayscaleFormat{ CL_DEPTH, CL_UNSIGNED_INT8 };
    // create input image object, which is read_only and has a format of RGBA + 8 bit depth
    cl::Image2D inputImage(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, rgbaFormat, width, height, 0, image.data()); // TODO: To be improved with CL_MEM_USE_HOST_PTR ?
    // create output image object, which is read and write so that it can be reused by next kernel; use format of grayscale + 8 bit depth
    cl::Image2D outputImageGray(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, grayscaleFormat, width, height);
    cl::Kernel kernelGrayscale(program, "convert_grayscale");

    // set arguments
    kernelGrayscale.setArg(0, inputImage);
    kernelGrayscale.setArg(1, outputImageGray);

    // create command queue with profiling enabled
    cl_command_queue_properties properties[]{ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl::CommandQueue queue(context, device, properties);
    cl::Event profEvent;

    // queue the kernel
    queue.enqueueNDRangeKernel(kernelGrayscale, cl::NullRange, cl::NDRange(width, height), cl::NullRange, 0, &profEvent);
    profEvent.wait();

    // print profiling
    double runTime = (double)(profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "Grayscale conversion execution time in microseconds " << runTime / (float)10e3 << std::endl;

    return outputImageGray;
}

cl::Buffer EnqueueResizeImage(cl::Context& context, cl::Program program, cl::Device device, cl::Image2D image, unsigned int width, unsigned int height, unsigned int resizeFactor, cl::CommandQueue& queue)
{
    // create output buffer object for image so that it can be accessed easier in the next part of the pipeline
    // read_write access given, so that buffer can be reused as input
    cl::Buffer outputImageBuffResized(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(unsigned char) * (width * height));
    cl::Kernel kernelResize(program, "resize_image");

    // set arguments
    kernelResize.setArg(0, resizeFactor);
    kernelResize.setArg(1, image);
    kernelResize.setArg(2, outputImageBuffResized);

    // create command queue with profiling enabled
    cl_command_queue_properties properties[]{ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl::CommandQueue queueResize(context, device, properties);
    cl::Event profEvent;

    // queue the resizing kernel
    queueResize.enqueueNDRangeKernel(kernelResize, cl::NullRange, cl::NDRange(width, height), cl::NullRange, 0, &profEvent);
    profEvent.wait();

    double runTime = (double)(profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    std::cout << "Resize execution time in microseconds " << runTime / (float)10e3 << std::endl;

    queue = queueResize;
    return outputImageBuffResized;
}


// Apply ZNCC algorithm for a given window size and max disparity
void EnqueuZNCC(const std::vector<unsigned char>& leftImage,
    const std::vector<unsigned char>& rightImage,
    int width, int height,
    int windowSize, int maxDisparity,
    std::vector<int>& disparityMap,
    char isLeftImage = 1
)
{
    int imgSize = width * height;

    int halfWindowSize = (windowSize - 1) / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int bestDisp = 0;
            float bestZNCC = -100.0;
            bool isBorderPixel = false;

            // handle borders | keep bestDisp at 0, so borders will be black
            if (y >= height - halfWindowSize || x >= width - halfWindowSize ||
                y <= halfWindowSize || x <= halfWindowSize)
            {
                isBorderPixel = true;
            }

            if (!isBorderPixel)
            {
                for (int d = 0; d < maxDisparity; d++)
                {
                    float zncc = 0.0;
                    float numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
                    float leftMean = 0.0, rightMean = 0.0;

                    // calculate mean for each window - future kernel - changes for different disparities, as the rightmean is calculated based on the disparity
                    int avgCount = 0;
                    for (int winY = -halfWindowSize; winY < halfWindowSize; winY++)
                    {
                        for (int winX = -halfWindowSize; winX < halfWindowSize; winX++)
                        {
                            // don't allow pixel to go to previous row
                            if (d > x + winX)
                            {
                                continue;
                            }

                            int leftPixelIndex = (y + winY) * width + (x + winX);
                            int rightPixelIndex = (y + winY) * width + (x + winX - isLeftImage * d);
                            if (rightPixelIndex >= imgSize ||
                                rightPixelIndex <= 0)
                            {
                                continue;
                            }

                            leftMean += leftImage[leftPixelIndex];
                            rightMean += rightImage[rightPixelIndex];
                            avgCount++;
                        }

                    }
                    leftMean = leftMean / avgCount;
                    rightMean = rightMean / avgCount;

                    for (int winY = -halfWindowSize; winY < halfWindowSize; winY++)
                    {
                        for (int winX = -halfWindowSize; winX < halfWindowSize; winX++)
                        {
                            // don't allow pixel to go to previous row
                            if (d > x + winX)
                            {
                                continue;
                            }

                            int leftPixelIndex = (y + winY) * width + (x + winX);
                            int rightPixelIndex = (y + winY) * width + (x + winX - isLeftImage * d);
                            if (rightPixelIndex >= imgSize ||
                                rightPixelIndex <= 0)
                            {
                                continue;
                            }

                            // calculate zncc value for each window
                            numerator += (leftImage[leftPixelIndex] - leftMean) * (rightImage[rightPixelIndex] - rightMean);
                            denominator1 += pow(leftImage[leftPixelIndex] - leftMean, 2);
                            denominator2 += pow(rightImage[rightPixelIndex] - rightMean, 2);

                        }

                    }

                    float denominator = sqrt(denominator1) * sqrt(denominator2);
                    if (denominator == 0) {
                        break;
                    }

                    zncc = numerator / denominator;
                    if (zncc > bestZNCC)
                    {
                        bestZNCC = zncc;
                        bestDisp = d;
                    }
                }
            }

            disparityMap[y * width + x] = bestDisp;
        }
    }
}

void CrossCheck(const std::vector<int>& dispMapLeft, const std::vector<int>& dispMapRight, const int& width, const int& height, const int& crossDiff, std::vector<int>& crossDispMap)
{
    // Loop over all pixels inside the image boundary
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // Get the disparity values for the current pixel in both directions
            int dispLeft = dispMapLeft[y * width + x];
            int dispRight = dispMapRight[y * width + x];

            // Check if the disparity values agree | abs used to account for rounding errors
            if (std::abs(dispLeft - dispRight) <= crossDiff) {
                // If the disparities agree, use the left disparity value as the final disparity for the pixel
                crossDispMap[y * width + x] = dispLeft;
            }
            else {
                // Otherwise, mark the pixel as invalid
                crossDispMap[y * width + x] = 0;
            }
        }
    }
}

void OcclusionFilling(const std::vector<int>& dispMap, const int& width, const int& height, const int& nCount, std::vector<int>& dispMapFilled)
{
    // Copy the input disparity map to the output disparity map
    std::copy(dispMap.begin(), dispMap.end(), dispMapFilled.begin());

    // Loop over all pixels inside the image boundary
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // handle borders | keep bestDisp at 0, so borders will stay black
            if (y >= height - (nCount / 2) || x >= width - (nCount / 2) ||
                y <= nCount / 2 || x <= nCount / 2)
            {
                continue;
            }

            // Check if the current pixel is marked as invalid
            if (dispMap[y * width + x] == 0) {

                // Initialize the list of valid disparity values in the n-neighborhood of the current pixel
                std::vector<int> neighbors;

                // Loop over the n-neighbors of the current pixel
                for (int dy = -nCount / 2; dy <= nCount / 2; dy++) {
                    for (int dx = -nCount / 2; dx <= nCount / 2; dx++) {
                        // Skip the center pixel
                        if (dx == 0 && dy == 0) continue;

                        // Get the disparity value for the current neighbor
                        int neighbor_disp = dispMap[(y + dy) * width + (x + dx)];

                        // If the neighbor is valid, add its disparity value to the list
                        if (neighbor_disp > 0) {
                            neighbors.push_back(neighbor_disp);
                        }
                    }
                }

                // If at least one valid disparity value was found in the n-neighborhood,
                // set the current pixel's disparity value to the median of the valid values
                if (!neighbors.empty()) {
                    dispMapFilled[y * width + x] = neighbors[neighbors.size() / 2];
                }
            }
        }
    }
}

void NormalizeToChar(const std::vector<int>& dispMap, const int& width, const int& height, const int& ndisp, std::vector<unsigned char>& normVec)
{
    // Loop over all pixels and normalize the disparity values
    for (int i = 0; i < width * height; i++) {
        normVec[i] = static_cast<unsigned char>(static_cast<float>(dispMap[i]) / ndisp * 255);
    }
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

    const char* depthmapOut = "../img/cl_depthmap_left.png";
    const char* depthmapOutRight = "../img/cl_depthmap_right.png";

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
        // Create source
        // Note: Both kernels are stored in one kernel file as
        // OpenCL C++ seems to have an issue if you pass a second kernel file
        std::ifstream kernelFile("../kernels/zncc_kernels.cl");
        std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

        cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

        // create context and build program
        cl::Context context(devices);
        cl::Program program(context, sources);

        program.build("-cl-std=CL1.2");

        // Kernel logic
        //// Grayscale conversion
        std::cout << "Converting left image to grayscale..." << std::endl;
        auto outputImageGrayLeft = EnqueueGrayScaleConversion(context, program, device, leftImage, width, height);
        std::cout << "Converting right image to grayscale..." << std::endl;
        auto outputImageGrayRight = EnqueueGrayScaleConversion(context, program, device, rightImage, width, height);

        //// Rescaling
        // update values depending on resolution
        int oldWidth = width;
        width = width / resizeFactor;
        height = height / resizeFactor;
        ndisp = ndisp * (static_cast<float>(width) / oldWidth);

        // enqueue resizing
        cl::CommandQueue queueResizeLeft, queueResizeRight;
        std::cout << "Resizing left image to 1/16 size..." << std::endl;
        auto outputImageResizedLeft = EnqueueResizeImage(context, program, device, outputImageGrayLeft, width, height, resizeFactor, queueResizeLeft);
        std::cout << "Resizing right image to 1/16 size..." << std::endl;
        auto outputImageResizedRight = EnqueueResizeImage(context, program, device, outputImageGrayRight, width, height, resizeFactor, queueResizeRight);

        // read the rescaled image output and put it into a vector
        cl::Event readEvent;
        std::vector<unsigned char> leftImageResized(width * height);
        queueResizeLeft.enqueueReadBuffer(outputImageResizedLeft, CL_TRUE, 0, sizeof(unsigned char) * leftImageResized.size(), leftImageResized.data(), 0, &readEvent);

        // print profiling
        double transferTime = (double)(readEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - readEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        std::cout << "Left resize read bus transfer time in microseconds " << transferTime / (float)10e3 << std::endl;

        error = lodepng::encode(depthmapOut, leftImageResized, width, height, LCT_GREY, 8);
        if (error) std::cout << "encoder error left image: " << error << ": " << lodepng_error_text(error) << std::endl;


        std::vector<unsigned char> rightImageResized(width * height);
        queueResizeRight.enqueueReadBuffer(outputImageResizedRight, CL_TRUE, 0, sizeof(unsigned char) * rightImageResized.size(), rightImageResized.data(), 0, &readEvent);

        // print profiling
        transferTime = (double)(readEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() - readEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());
        std::cout << "Right resize read bus transfer time in microseconds " << transferTime / (float)10e3 << std::endl;

        error = lodepng::encode(depthmapOutRight, rightImageResized, width, height, LCT_GREY, 8);
        if (error) std::cout << "encoder error right image: " << error << ": " << lodepng_error_text(error) << std::endl;

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






    //// start timing execution time
    //LARGE_INTEGER start, end, frequency;
    //double elapsed_time;

    //QueryPerformanceFrequency(&frequency);

    //QueryPerformanceCounter(&start);

    //// convert image to grayscale, ignoring the alpha channel
    //std::vector<unsigned char> leftImageGray(width * height);
    //std::vector<unsigned char> rightImageGray(width * height);
    //GrayScaleConversion(leftImage, width, height, leftImageGray);
    //GrayScaleConversion(rightImage, width, height, rightImageGray);

    //// resize image
    //std::vector<unsigned char> leftImageResized(width * height);
    //std::vector<unsigned char> rightImageResized(width * height);
    //ResizeImage(leftImageGray, width, height, resizeFactor, leftImageResized);
    //ResizeImage(rightImageGray, width, height, resizeFactor, rightImageResized);

    //// update values depending on resolution
    //int oldWidth = width;
    //width = width / resizeFactor;
    //height = height / resizeFactor;
    //ndisp = ndisp * (static_cast<float>(width) / oldWidth);

    //// apply zncc
    //std::vector<int> leftImageDisparity(width * height);
    //std::vector<int> rightImageDisparity(width * height);
    //CalcZNCC(leftImageResized, rightImageResized, width, height, win_size, ndisp, leftImageDisparity);
    //CalcZNCC(rightImageResized, leftImageResized, width, height, win_size, ndisp, rightImageDisparity, -1);

    //// CrossChecking
    //std::vector<int> crossCheckedMap(width * height);
    //CrossCheck(leftImageDisparity, rightImageDisparity, width, height, crossDiff, crossCheckedMap);

    //// occlusion filling
    //std::vector<int> oclussionFilledMap(width * height);
    //OcclusionFilling(crossCheckedMap, width, height, neighbours, oclussionFilledMap);

    //// normalization to 8 bit
    //std::vector<unsigned char> depthmapNormalized(width * height);
    //NormalizeToChar(oclussionFilledMap, width, height, ndisp, depthmapNormalized);

    //// end execution timing and print
    //QueryPerformanceCounter(&end);

    //elapsed_time = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    //std::cout << "Elapsed time: " << elapsed_time << " seconds\n";

    //std::cout << "Elapsed time: " << elapsed_time / 60 << " minutes\n";

    //// encode resized and grayscaled images (im*_out)
    //error = lodepng::encode(depthmapOut, depthmapNormalized, width, height, LCT_GREY, 8);
    //if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;
}