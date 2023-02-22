#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <lodepng.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <fstream>
#include <tuple>
#include <iostream>
#include <chrono>

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
	const char* imgName = "img/im0.png";
	const char* imgNameOut = "img/imCV_out.png";

	std::vector<unsigned char> img;
	std::vector<unsigned char> imgOut;
	std::vector<unsigned char> imgFiltered;
	unsigned int width, height;

	// decode images

	auto start = std::chrono::high_resolution_clock::now();
	unsigned int error = lodepng::decode(img, width, height, imgName, LCT_RGBA, 8);
	auto stop = std::chrono::high_resolution_clock::now();
	if (error) std::cout << "decoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Loading image execution time in microseconds: " << duration.count() << std::endl;

	imgOut.resize(width * height);
	std::fill(imgOut.begin(), imgOut.end(), 0);

	imgFiltered.resize(width * height);
	std::fill(imgFiltered.begin(), imgFiltered.end(), 0);

	try
	{
		// create the program
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		auto platform = platforms.front();

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		auto device = devices.front();


		std::ifstream grayscale_kernel_file("kernels/image_manipulator_kernels.cl");
		std::string grayscale_src(std::istreambuf_iterator<char>(grayscale_kernel_file), (std::istreambuf_iterator<char>()));

		// Note: OpenCL C++ seems to have an issue if you pass a second kernel file
		cl::Program::Sources sources(1, std::make_pair(grayscale_src.c_str(), grayscale_src.length() + 1));

		cl_int err = 0;
		cl::Context context(devices);

		cl::Program program(context, sources, &err);

		program.build("-cl-std=CL1.2");


		// Kernel logic
		// 
		// Grayscale conversion
		cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * img.size(), img.data(), &err);
		cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned char) * imgOut.size(), nullptr, &err);
		cl::Kernel kernel_grayscale(program, "convert_grayscale");

		err = kernel_grayscale.setArg(0, inBuf);
		err = kernel_grayscale.setArg(1, outBuf);

		cl_command_queue_properties properties[]{ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		cl::CommandQueue queue(context, device, properties, &err);
		cl::Event prof_event;

		err = queue.enqueueNDRangeKernel(kernel_grayscale, cl::NullRange, cl::NDRange(img.size()), cl::NullRange, 0, &prof_event);
		prof_event.wait();
		err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(unsigned char) * imgOut.size(), imgOut.data()); //blocking so that we can encode full image

		double run_time = (double)(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		std::cout << "Grayscale conversion execution time in microseconds " << run_time / (float)10e3 << std::endl;






		// Moving filter
		cl::Kernel kernel_filter(program, "apply_moving_filter");

		cl::Buffer img_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * imgOut.size(), imgOut.data(), &err);
		cl::Buffer filter_Buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * (GAUSSIAN_SIZE * GAUSSIAN_SIZE), (void*)&gaussian_filter_matrix[0], &err);
		cl::Buffer filtered_Buf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(unsigned char) * imgFiltered.size(), nullptr, &err);

		kernel_filter.setArg(0, height);
		kernel_filter.setArg(1, width);
		kernel_filter.setArg(2, GAUSSIAN_SIZE);
		kernel_filter.setArg(3, img_Buf);
		kernel_filter.setArg(4, filter_Buf);
		kernel_filter.setArg(5, filtered_Buf);

		cl::CommandQueue queue_grayscale(context, device, properties, &err);

		err = queue_grayscale.enqueueNDRangeKernel(kernel_filter, cl::NullRange, cl::NDRange(height, width), cl::NullRange, 0, &prof_event);
		prof_event.wait();
		err = queue_grayscale.enqueueReadBuffer(filtered_Buf, CL_TRUE, 0, sizeof(unsigned char) * imgFiltered.size(), imgFiltered.data()); //blocking so that we can encode full image

		run_time = (double)(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		std::cout << "Gaussian moving filter execution time in microseconds " << run_time / (float)10e3 << std::endl;

		// encode blurred images
		start = std::chrono::high_resolution_clock::now();
		error = lodepng::encode(imgNameOut, imgFiltered, width, height, LCT_GREY, 8);
		stop = std::chrono::high_resolution_clock::now();
		if (error) std::cout << "encoder error first image: " << error << ": " << lodepng_error_text(error) << std::endl;

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

	return 1;
}