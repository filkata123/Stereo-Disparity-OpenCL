#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <lodepng.h>

#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>

// gaussian kernel as seen in https://www.codingame.com/playgrounds/2524/basic-image-manipulation/filtering
const int GAUSSIAN_SIZE = 5;
const float gaussian_filter_matrix[GAUSSIAN_SIZE * GAUSSIAN_SIZE] =
{
	1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f,
	4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
	6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f,
	4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
	1 / 256.0f, 4 / 256.0f, 6 / 256.0f, 4 / 256.0f, 1 / 256.0f
};

int main()
{
	// set input and output files
	const char* imgName = "img/im0.png";
	const char* imgNameOut = "img/imCV_out.png";

	// create containers for the different steps of the image processing
	std::vector<unsigned char> img;
	std::vector<unsigned char> imgOut;
	std::vector<unsigned char> imgFiltered;
	unsigned int width, height;

	// decode images and print out time it took to do so
	auto start = std::chrono::high_resolution_clock::now();
	unsigned int error = lodepng::decode(img, width, height, imgName, LCT_RGBA, 8);
	auto stop = std::chrono::high_resolution_clock::now();
	if (error) std::cout << "decoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Loading image execution time in microseconds: " << duration.count() << std::endl;

	// initialize ouput vectors to 0 with the newly acquired resolution (width * height)
	imgOut.resize(width * height);
	std::fill(imgOut.begin(), imgOut.end(), 0);

	imgFiltered.resize(width * height);
	std::fill(imgFiltered.begin(), imgFiltered.end(), 0);

	// OpenCL part
	try
	{
		// create the program
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		auto platform = platforms.front();

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		auto device = devices.front();

		// Both kernels are stored in one kernel file as
		// Note: OpenCL C++ seems to have an issue if you pass a second kernel file
		std::ifstream kernel_file("kernels/image_manipulator_kernels.cl");
		std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

		cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));


		// create context and build program
		cl::Context context(devices);

		cl::Program program(context, sources);

		program.build("-cl-std=CL1.2");


		// Kernel logic
		//// Grayscale conversion

		// create device read_only input buffer, which copies img data to kernel
		cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * img.size(), img.data());
		// create device write_only output buffer
		cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned char) * imgOut.size(), nullptr);
		cl::Kernel kernel_grayscale(program, "convert_grayscale");

		// set arguments
		kernel_grayscale.setArg(0, inBuf);
		kernel_grayscale.setArg(1, outBuf);

		// create command queue with profiling enabled
		cl_command_queue_properties properties[]{ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		cl::CommandQueue queue(context, device, properties);
		cl::Event prof_event;

		// queue the kernel
		queue.enqueueNDRangeKernel(kernel_grayscale, cl::NullRange, cl::NDRange(img.size()), cl::NullRange, 0, &prof_event);
		prof_event.wait();

		// queue the read, which puts the grayscale raw image into the new vector
		cl::Event read_event;
		queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(unsigned char) * imgOut.size(), imgOut.data(), 0, &read_event); //blocking so that we can encode full image

		// print profiling
		double run_time = (double)(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
		double transfer_time = (double)(read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		std::cout << "Grayscale conversion execution time in microseconds " << run_time / (float)10e3 << std::endl;
		std::cout << "Grayscale conversion read bus transfer time in microseconds " << transfer_time / (float)10e3 << std::endl;

		//// Moving filter
		// select second kernel
		cl::Kernel kernel_filter(program, "apply_moving_filter");

		// create device read_only input buffer, which copies grayscale img data to kernel
		cl::Buffer img_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * imgOut.size(), imgOut.data());
		// create device read_only filter buffer, which copies filter to kernel
		cl::Buffer filter_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * (GAUSSIAN_SIZE * GAUSSIAN_SIZE), (void*)&gaussian_filter_matrix[0]);
		// create device write_only output buffer
		cl::Buffer filtered_Buf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned char) * imgFiltered.size(), nullptr);

		// set arguments
		kernel_filter.setArg(0, height);
		kernel_filter.setArg(1, width);
		kernel_filter.setArg(2, GAUSSIAN_SIZE);
		kernel_filter.setArg(3, img_Buf);
		kernel_filter.setArg(4, filter_Buf);
		kernel_filter.setArg(5, filtered_Buf);

		// create command queue with profiling enabled (properties)
		cl::CommandQueue queue_filter(context, device, properties);

		// queue kernel as 2D image
		queue_filter.enqueueNDRangeKernel(kernel_filter, cl::NullRange, cl::NDRange(height, width), cl::NullRange, 0, &prof_event);
		prof_event.wait();
		// queue read, which will put filtered img into vector. this 
		// blocking so that we can encode full image
		queue_filter.enqueueReadBuffer(filtered_Buf, CL_TRUE, 0, sizeof(unsigned char) * imgFiltered.size(), imgFiltered.data(), 0, &read_event);

		// print profiling
		run_time = (double)(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());
		transfer_time = (double)(read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		std::cout << "Gaussian moving filter execution time in microseconds " << run_time / (float)10e3 << std::endl;
		std::cout << "Gaussian moving filter read bus transfer time in microseconds " << transfer_time / (float)10e3 << std::endl;

		// encode blurred images
		start = std::chrono::high_resolution_clock::now();
		error = lodepng::encode(imgNameOut, imgFiltered, width, height, LCT_GREY, 8);
		stop = std::chrono::high_resolution_clock::now();
		if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

		// print more profiling
		duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Saving image execution time in microseconds: " << duration.count() << std::endl;

		std::cout << "------------HARDWARE INFORMATION------------" << std::endl;
		auto platName = platform.getInfo<CL_PLATFORM_NAME>();
		auto devVersion = device.getInfo<CL_DEVICE_VERSION>();
		auto devInfo = device.getInfo<CL_DEVICE_NAME>();
		auto devDriver = device.getInfo<CL_DRIVER_VERSION>();
		auto devCVersion = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
		auto devPCunits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		auto devWorkItemDim = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();

		std::cout << "Platform count: " << platforms.size() << std::endl;
		std::cout << "Platform 1 name: " << platName << std::endl;
		std::cout << "Device count on platform 1: " << devices.size() << std::endl;
		std::cout << "Device information: " << devInfo << std::endl;
		std::cout << "Hardware version: " << devVersion << std::endl;
		std::cout << "Driver version: " << devDriver << std::endl;
		std::cout << "OpenCL C version: " << devCVersion << std::endl;
		std::cout << "Parallel Compute units: " << devPCunits << std::endl;
		std::cout << "Max Work Item Dimensions: " << devWorkItemDim << std::endl;

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