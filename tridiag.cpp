

// Tridiagonal solve using PCR

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <CL/cl.h>     
#include"OpenCLUtils.h"


// Because we are working with tridiagonal matrices LOCAL_WORK_SIZE 
// can be as large as 256. 

#define N 524288				/* N is the total number of entries for ALL systems */
#define LOCAL_WORK_SIZE 512		/* This is the size of the systems (number of entries on main diagonal) */
int log2(int n)
{
	int res = 0;
	while (n > 1) { n >>= 1; res++; }
	return res;
}
/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
	int i, j, k;
	const unsigned int number_of_systems = N / LOCAL_WORK_SIZE;
	const unsigned int mem_size = sizeof(float)*N;

	// allocate host arrays.
	float *a = (float*)malloc(mem_size);
	float *b = (float*)malloc(mem_size);
	float *c = (float*)malloc(mem_size);
	float *d = (float*)malloc(mem_size);
	float *x = (float*)malloc(mem_size);

	// initialise the host memory
	// b is the main diagonal
	// a and c are the sub and super diagonals
	for (i = 0; i < number_of_systems; i++)
	{
		for (j = 0; j < LOCAL_WORK_SIZE; j++)
		{
			a[i*LOCAL_WORK_SIZE +j] = -1.0f;
			b[i*LOCAL_WORK_SIZE + j] =  2.0f;
			c[i*LOCAL_WORK_SIZE + j] = -1.0f;
			d[i*LOCAL_WORK_SIZE + j] = 1.0*sin(1.0f*j/LOCAL_WORK_SIZE);
			x[i*LOCAL_WORK_SIZE + j] = 0.0f;
		}
		a[i*LOCAL_WORK_SIZE] = 0.0f;
		c[i*LOCAL_WORK_SIZE + LOCAL_WORK_SIZE - 1] = 0.0f;
	}


	// OpenCL specific variables
	cl_device_id device;
	cl_context clGPUContext;
	cl_command_queue clCommandQue;
	cl_program clProgram;
	cl_kernel clKernel;


	cl_event ev;



	size_t dataBytes;
	size_t maxKernelWorkSize;
	size_t preferredKernelWorkSize;
	cl_int errcode;

	// OpenCL device memory for matrices
	cl_mem device_a;
	cl_mem device_b;
	cl_mem device_c;
	cl_mem device_d;
	cl_mem device_x;

	/*****************************************/
	/* Initialize OpenCL */
	/*****************************************/

	/* Create a device and context */
	device = create_device();
	clGPUContext = clCreateContext(NULL, 1, &device, NULL, NULL, &errcode);
	if (errcode < 0) {
		perror("Couldn't create a context");
		exit(1);
	}


	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &errcode);
	if (errcode < 0) {
		perror("Couldn't create a command que");
		exit(1);
	}

	// Setup device memory
	/* We are passing the host memory as an argument. This is where
	the device memory obtains the data from the host memory. */
	device_a = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, a, &errcode);
	if (errcode < 0) {
		perror("Couldn't create a buffer: device_a");
		exit(1);
	}
	device_b = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, b, &errcode);
	if (errcode < 0) {
		perror("Couldn't create a buffer: device_b");
		exit(1);
	}
	device_c = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, c, &errcode);
	if (errcode < 0) {
		perror("Couldn't create a buffer: device_c");
		exit(1);
	}
	device_d = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N, d, &errcode);
	if (errcode < 0) {
		perror("Couldn't create a buffer: device_d");
		exit(1);
	}
	device_x = clCreateBuffer(clGPUContext, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, &errcode);
	if (errcode < 0) {
		perror("Couldn't create a buffer: device_x");
		exit(1);
	}

	// Obtain size of source file
	FILE* fp = fopen("ModifiedPCR.cl", "rb");
	fseek(fp, 0, SEEK_END);
	const size_t lSize = ftell(fp);
	rewind(fp);

	// Read file content into buffer
	unsigned char* buffer = (unsigned char*)malloc(lSize + 1);
	buffer[lSize] = '\0';
	fread(buffer, sizeof(char), lSize, fp);
	fclose(fp);

	//create program from buffer
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char**)&buffer, &lSize, &errcode);
	errcode = clBuildProgram(clProgram, 1, &device, NULL, NULL, NULL);

	// Obtain the build log
	size_t len = 0;
	cl_int ret = CL_SUCCESS;
	ret = clGetProgramBuildInfo(clProgram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
	char *buffer1 = (char*)calloc(len, sizeof(char));
	ret = clGetProgramBuildInfo(clProgram, device, CL_PROGRAM_BUILD_LOG, len, buffer1, NULL);

	// Print the build log
	printf("%s", buffer1);

	clKernel = clCreateKernel(clProgram, "modified_pcr", &errcode);

	if (errcode < 0) {
		perror("Kernel creation failed");
		exit(1);
	}

	errcode = clGetKernelWorkGroupInfo(clKernel, device,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
		&maxKernelWorkSize, NULL);

	errcode = clGetKernelWorkGroupInfo(clKernel, device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
		&preferredKernelWorkSize, NULL);

	printf("CL_KERNEL_WORK_GROUP_SIZE: %d\n", maxKernelWorkSize);
	printf("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %d\n\n", preferredKernelWorkSize);



	size_t localWorkSize = LOCAL_WORK_SIZE;
	size_t globalWorkSize = N;
	int iterations = log2((int)localWorkSize / 2)-4;

	/* Set the arguments for the kernel. */
	errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&device_a);
	if (errcode < 0) {
		perror("Setting kernel argument failed: device_a");
		exit(1);
	}
	errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&device_b);
	errcode |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&device_c);
	errcode |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&device_d);
	errcode |= clSetKernelArg(clKernel, 4, sizeof(cl_mem), (void *)&device_x);
	errcode |= clSetKernelArg(clKernel, 5, (localWorkSize + 1) * 5 * sizeof(float), NULL);
	errcode |= clSetKernelArg(clKernel, 6, sizeof(int), &localWorkSize);
	errcode |= clSetKernelArg(clKernel, 7, sizeof(unsigned int), &number_of_systems);
	errcode |= clSetKernelArg(clKernel, 8, sizeof(int), &iterations);

	errcode = clEnqueueNDRangeKernel(clCommandQue,
		clKernel, 1, NULL, &globalWorkSize,
		&localWorkSize, 0, NULL, &ev);

	if (errcode < 0) {
		perror("enqueue failed");
		exit(1);
	}

	clWaitForEvents(1, &ev);

	clFinish(clCommandQue);
	cl_ulong time_start;
	cl_ulong time_end;
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	double nanoSeconds = time_end - time_start;
	printf("OpenCl Execution time is: %0.3f milliseconds \n", nanoSeconds / 1000000.0);



	errcode = clEnqueueReadBuffer(clCommandQue,
		device_x, CL_FALSE, 0, mem_size,
		x, 0, NULL, &ev);
/*
	clWaitForEvents(1, &ev);

	for (i = 0; i < N; i++)
	{
		printf("%f\n", x[i]);
	}
*/

	printf("\nFinished Calculation.\n");

	//  clean up memory
	free(a);
	free(b);
	free(c);
	free(d);
	free(x);


	clReleaseMemObject(device_a);
	clReleaseMemObject(device_b);
	clReleaseMemObject(device_c);
	clReleaseMemObject(device_d);
	clReleaseMemObject(device_x);

	clReleaseContext(clGPUContext);
	clReleaseKernel(clKernel);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(clCommandQue);
}
