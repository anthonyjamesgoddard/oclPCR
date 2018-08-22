#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include<CL\cl.h>

cl_device_id create_device() {

	cl_platform_id* platforms;
	cl_uint num_platforms,num_devices;
	cl_device_id* devices;
	char platform_name_data[50];
	char name_data[50];
	cl_int i,err,platformchoice;
	size_t ComputeUnitCount, MaxWorkGroupSize;
	size_t MaxWorkItemSizes[3] = { 0,0,0 };

	platformchoice =1;

	/* Find out how many platforms there are */
	err = clGetPlatformIDs(1, NULL, &num_platforms);
	if (err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	/* Reserve memory for platforms*/
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms);

	/* Obtain the available platforms and store them in the array platforms */
	clGetPlatformIDs(num_platforms, platforms, NULL);

	/* We want to know the names of the platforms.
	This will the inform us and lead to a
	cannonical choice for 'platformchoice'.*/

	for (i = 0; i < num_platforms; i++)
	{
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name_data), platform_name_data, NULL);
		if (err < 0)
		{
			perror("Unable to obtain information about platform");
		}
		printf("%s\n", platform_name_data);
	}

	printf("\nSearching %s for available devices...\n", platform_name_data);

	/* Obtain the number of GPUS available on this platform */
	err = clGetDeviceIDs(platforms[platformchoice], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	if (err == CL_DEVICE_NOT_FOUND) 
	{
		perror("No GPU devices available");
		exit(1);
	}
	if (err < 0) {
		perror("Could not access any devices. Not as a result of the device not being found. Debug for error code");
		exit(1);
	}
	/* Reserve memory for devices */
	devices = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);

	/* Populate devices with devices compatible with the chosen platform */
	clGetDeviceIDs(platforms[platformchoice], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

	for (i = 0; i < num_devices; i++)
	{
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name_data), name_data, NULL);
		err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &ComputeUnitCount, NULL);
		err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &MaxWorkGroupSize, NULL);
		err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(size_t), MaxWorkItemSizes, NULL);

		if (err < 0)
		{
			perror("Unable to obtain information about device");
		}
		printf("CL_DEVICE_NAME: %s\n", name_data);
		printf("CL_DEVICE_MAX_COMPUTE_UNITS: %d\n",ComputeUnitCount);
		printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n",MaxWorkGroupSize);
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: (%d,%d,%d)\n", MaxWorkItemSizes[0], MaxWorkItemSizes[1], MaxWorkItemSizes[2]);
	}
	return devices[0];
}

