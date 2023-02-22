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

	// flattened 2D array | can be done with nested std::vector
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

	// output array initialized to 0
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

		// get first platform and print it
		auto platform = platforms.front();
		auto platName = platform.getInfo<CL_PLATFORM_NAME>();
		std::cout << "Platform: " << platName << std::endl;

		// get GPU device and pritn information
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		auto device = devices.front();

		auto devVersion = device.getInfo<CL_DEVICE_VERSION>();
		auto devInfo = device.getInfo<CL_DEVICE_NAME>();

		std::cout << "Hardware version: " << devVersion << std::endl;
		std::cout << "Compute device name: " << devInfo << std::endl;


		// Add kernel as a source file
		std::ifstream passed_file("../kernels/add_matrix.cl");
		std::string src(std::istreambuf_iterator<char>(passed_file), (std::istreambuf_iterator<char>()));

		cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

		// Create context and build program
		cl::Context context(devices);
		cl::Program program(context, sources);
		program.build("-cl-std=CL1.2");

		// Kernel logic
		cl::Kernel kernel(program, "add_matrix");

		// Create two buffers for both input arrays. Both arrays should be readable by device and host.
		// The pointer to the beginning of the memory is coppied to the device
		cl::Buffer a_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (rows * cols), (void*)&A[0]);
		cl::Buffer b_in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * (rows * cols), (void*)&B[0]);

		// Create buffer for result. The device directly writes the result to host memory.
		cl::Buffer c_out(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * (rows * cols), (void*)&C[0]);

		// set arguments
		kernel.setArg(0, rows);
		kernel.setArg(1, cols);
		kernel.setArg(2, a_in);
		kernel.setArg(3, b_in);
		kernel.setArg(4, c_out);

		// create command queue, which has profiling enabled
		cl_command_queue_properties properties[]{ CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
		cl::CommandQueue queue(context, device, properties);
		cl::Event prof_event;

		// queue a command with correct size on the kernel (array is technically 1D, but it can be observed as 2D by kernel)
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(rows, cols), cl::NullRange,0, &prof_event);
		// wait until profiling is finished
		prof_event.wait();
		// queue a command to read a buffer object to host memory in C
		queue.enqueueReadBuffer(c_out, CL_TRUE, 0, sizeof(int) * (rows * cols), (void*)&C[0]);

		// calculate execution time w/ through profiling
		double run_time = (double)(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		std::cout << "------------------------------" << std::endl;
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


	// visualize arrays if necessary
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