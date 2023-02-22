#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <cstdlib>
#include <string>
#include <fstream>
#include <iostream>

int main()
{
	// Setup matrices
	int rows = 100;
	int cols = 100;
	srand(time(NULL));

	int* A = new int[rows * cols];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			A[i * cols + j] = rand() % 100; // Generate random number between 0 and 99
		}
	}

	int* B = new int[rows * cols];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			B[i * cols + j] = rand() % 100; // Generate random number between 0 and 99
		}
	}

	int* C = new int[rows * cols];
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			C[i * cols + j] = 0;
		}
	}

	try
	{
		// create the program
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		auto platform = platforms.front();
		auto platName = platform.getInfo<CL_PLATFORM_NAME>();

		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

		auto device = devices.front();
		auto devVendor = device.getInfo<CL_DEVICE_VENDOR>();
		auto devVersion = device.getInfo<CL_DEVICE_VERSION>();
		auto devInfo = device.getInfo<CL_DEVICE_NAME>();

		std::ifstream passed_file("../kernels/add_matrix.cl");
		std::string src(std::istreambuf_iterator<char>(passed_file), (std::istreambuf_iterator<char>()));

		cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

		cl_int err = 0;
		cl::Context context(devices);

		cl::Program program(context, sources, &err);

		program.build("-cl-std=CL1.2");


		// Kernel logic
		cl::Kernel kernel(program, "add_matrix");

		cl::Buffer a_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (rows * cols), (void*)&A[0]);
		cl::Buffer b_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (rows * cols), (void*)&B[0]);
		cl::Buffer c_out(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * (rows * cols), (void*)&C[0]);

		kernel.setArg(0, rows);
		kernel.setArg(1, cols);
		kernel.setArg(2, a_in);
		kernel.setArg(3, b_in);
		kernel.setArg(4, c_out);

		cl_command_queue_properties properties[]{ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		cl::CommandQueue queue(context, device, properties, &err);
		cl::Event prof_event;

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows, cols), cl::NullRange,0, &prof_event);
		prof_event.wait();
		queue.enqueueReadBuffer(c_out, CL_TRUE, 0, sizeof(int) * (rows * cols), (void*)&C[0]);


		double run_time = (double)(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		std::cout << "Compute device information: " << devInfo << std::endl;
		std::cout << "Execution time in microseconds " << run_time / (float)10e3 << std::endl;

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

	/*std::cout << "Matrix A:";
	for (int i = 0; i < rows * cols; i++) {
		std::cout << A[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "Matrix B:";
	for (int i = 0; i < rows * cols; i++) {
		std::cout << B[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "Matrix C:";
	for (int i = 0; i < rows * cols; i++) {
			std::cout << C[i] << " ";
	}
	std::cout << std::endl;*/
	
	// Free memory
	delete[] A;
	delete[] B;
	delete[] C;
}