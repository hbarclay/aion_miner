
#include <CL/cl.hpp>
#include <errno.h>
#include <iostream>
#include "opencl_solver.h"
#include "_kernel.h"

namespace {

unsigned scan_platform(cl_platform_id plat, cl_uint *nr_devs_total,
	cl_platform_id *plat_id, cl_device_id *dev_id, int device_id) {
    cl_device_type	typ = CL_DEVICE_TYPE_ALL;
    cl_uint		nr_devs = 0;
    cl_device_id	*devices;
    cl_int		status;
    unsigned		found = 0;
    unsigned		i;

    status = clGetDeviceIDs(plat, typ, 0, NULL, &nr_devs);

    // With multiple platforms, valid devices may not be on current platform
    if (status == CL_DEVICE_NOT_FOUND){
		throw std::runtime_error("Device not found: clGetDeviceIDs " + std::to_string(status));
    }
    else if (status != CL_SUCCESS)
		throw std::runtime_error("clGetDeviceIDs " + std::to_string(status));

    if (nr_devs == 0)
		return 0;

    devices = (cl_device_id *)malloc(nr_devs * sizeof (*devices));
    status = clGetDeviceIDs(plat, typ, nr_devs, devices, NULL);

    if (status != CL_SUCCESS)
		throw std::runtime_error("clGetDeviceIDs " + std::to_string(status));
    i = 0;
    while (i < nr_devs) {
		if (*nr_devs_total == device_id) {
			found = 1;
			*plat_id = plat;
			*dev_id = devices[i];
			break;
		}
		(*nr_devs_total)++;
		i++;
    }
    free(devices);
    return found;
}

bool scan_platforms(cl_platform_id *plat_id, cl_device_id *dev_id, int device_id) {
	bool found = false;
	cl_uint nr_platforms;
    cl_platform_id	*platforms;
    cl_uint	i, nr_devs_total;
    cl_int	status;

    status = clGetPlatformIDs(0, NULL, &nr_platforms);
    
	if (status != CL_SUCCESS)
		throw std::runtime_error("Cannot get OpenCL platforms" + std::to_string(status));
    if (!nr_platforms)
		exit(1);
    
	platforms = (cl_platform_id *)malloc(nr_platforms * sizeof (*platforms)); 
	if (!platforms)
		throw std::runtime_error("malloc: " + std::to_string(errno));

    status = clGetPlatformIDs(nr_platforms, platforms, NULL);
    if (status != CL_SUCCESS)
		throw std::runtime_error("clGetPlatformIDs " + std::to_string(status));

    i = nr_devs_total = 0;
    while (i < nr_platforms) {
		if (scan_platform(platforms[i], &nr_devs_total, plat_id, dev_id, device_id)) {
			found = true;
			break;
		}
		i++;
    }
    
    free(platforms);
	return found;
}

void get_program_build_log(cl_program program, cl_device_id device)
{
    cl_int		status;
    char	        val[2*1024*1024];
    size_t		ret = 0;
    status = clGetProgramBuildInfo(program, device,
	    CL_PROGRAM_BUILD_LOG,
	    sizeof (val),	// size_t param_value_size
	    &val,		// void *param_value
	    &ret);		// size_t *param_value_size_ret
    if (status != CL_SUCCESS)
		throw std::runtime_error("clGetProgramBuildInfo " + std::to_string(status));
    std::cerr << val << std::endl;
}

}  // namespace 

void opencl_solver::getinfo(int plat_id, int dev_id, 
									std::string& gpu_name,
									int& cu_count,
									std::string& version) {

}

std::string opencl_solver::getdevinfo() {

}

void opencl_solver::print_opencl_info(std::ostream& os) {
	std::vector<cl::Platform> platforms;  
	cl::Platform::get(&platforms);  


	int platform_id = 0;
	int device_id = 0;

	os << "Number of Platforms: " << platforms.size() << std::endl;

	for(std::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it){
		cl::Platform platform(*it);

		os << "Platform ID: " << platform_id++ << std::endl;  
		os << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;  
		os << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;  

		std::vector<cl::Device> devices;  
		platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);  

		for(std::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2){
			cl::Device device(*it2);

			os << "\tDevice " << device_id++ << ": " << std::endl;
			os << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;  
			os << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
			os << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;  
			os << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
			os << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
			os << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
			os << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
			os << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
			os << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
			os << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
		}  
		os<< std::endl;
	} 
}

// find device, create command queues and build kernel source
void opencl_solver::start(opencl_solver& device_context) {
	cl_platform_id plat_id = 0;
	cl_device_id dev_id = 0;
	if (!scan_platforms(&plat_id, &dev_id, device_context.platform_id)) {
		throw std::runtime_error("Device number not found");
	}
	cl_int status;

	cl_context context = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &status);

	if (status != CL_SUCCESS)	
		throw std::runtime_error("clCreateContext " + std::to_string(status));

	cl_command_queue queue = clCreateCommandQueue(context, dev_id, 0, &status);
	if (status != CL_SUCCESS)
		throw std::runtime_error("clCreateCommandQueue " + std::to_string(status));

	const char* source = ocl_code;
	size_t source_len = strlen(ocl_code);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_len, &status);
	
	if (status != CL_SUCCESS || !program)
		throw std::runtime_error("clCreateProgramWithSource " + std::to_string(status));

	status = clBuildProgram(program, 1, &dev_id, "", NULL, NULL);
	if (status != CL_SUCCESS) {
		get_program_build_log(program, dev_id);
		throw std::runtime_error("OpenCL build failed");
	}

}

// close command queues, etc
void opencl_solver::stop(opencl_solver& device_context) {

}

// call kernels and generate solutions
void opencl_solver::solve(const char *tequihash_header,
		unsigned int tequihash_header_len, const char* nonce,
		unsigned int nonce_len, std::function<bool()> cancelf,
		std::function<
				void(const std::vector<uint32_t>&, size_t,
						const unsigned char*)> solutionf,
		std::function<void(void)> hashdonef,
		opencl_solver& device_context) {

}


