
#include <CL/cl.hpp>
#include <errno.h>
#include <iostream>
#include <assert.h>
#include "opencl_solver.h"
#include "_kernel.h"
// FIXME temporary fix for build error
typedef unsigned char uchar;
#include "param.h"
#include "blake2b.h"

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
	char			val[2*1024*1024];
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

void errClSetKernelArg(cl_kernel k, cl_uint argnum, cl_mem *a) {
	cl_int status;
	status = clSetKernelArg(k, argnum, sizeof(*a), a);
	if (status != CL_SUCCESS)
		throw std::runtime_error("clSetKernelArg " + std::to_string(status));
}

cl_mem errClCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr) {
	cl_int status;
	cl_mem ret;
	ret = clCreateBuffer(context, flags, size, host_ptr, &status);
	if (status != CL_SUCCESS || !ret)
		throw std::runtime_error("clCreateBuffer " + std::to_string(status));
	return ret;
}

void errClEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events, const cl_event* event_wait_list, cl_event* event) {
	cl_uint status;

	status = clEnqueueNDRangeKernel(q, k, work_dim, global_work_offset, global_work_size, local_work_size, num_events, event_wait_list, event);
	if (status != CL_SUCCESS)
		throw std::runtime_error("clEnqueueNDRangeKernel " + std::to_string(status));
}

void initializeHashTable(cl_command_queue queue, cl_kernel k_init_ht, cl_mem buf_ht, cl_mem rowCounters) {
	size_t global_ws = NR_ROWS / ROWS_PER_UINT;
	size_t local_ws = 256;
	cl_int status;

	status = clSetKernelArg(k_init_ht, 0, sizeof(buf_ht), &buf_ht);
	clSetKernelArg(k_init_ht, 1, sizeof(rowCounters), &rowCounters);
	if (status != CL_SUCCESS)
		throw std::runtime_error("clSetKernelArg k_init_ht " + std::to_string(status));
	
	errClEnqueueNDRangeKernel(queue, k_init_ht,
		1,
		NULL,
		&global_ws,
		&local_ws,
		0, 
		NULL,
		NULL);
}

size_t selectWorkSize() {
	// FIXME compute units
	size_t numComputeUnits = 36;
	size_t worksize = 64 * BLAKE_WPS * 4 * 36;
	while (NR_INPUTS % worksize)
		worksize += 64;
	return worksize;
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

	device_context.clcontext = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &status);

	if (status != CL_SUCCESS)	
		throw std::runtime_error("clCreateContext " + std::to_string(status));

	device_context.command_queue = clCreateCommandQueue(device_context.clcontext, dev_id, 0, &status);
	if (status != CL_SUCCESS)
		throw std::runtime_error("clCreateCommandQueue " + std::to_string(status));

	const char* source = ocl_code;
	size_t source_len = strlen(ocl_code);

	device_context.program = clCreateProgramWithSource(device_context.clcontext, 1, (const char**)&source, &source_len, &status);
	
	if (status != CL_SUCCESS || !device_context.program)
		throw std::runtime_error("clCreateProgramWithSource " + std::to_string(status));

	status = clBuildProgram(device_context.program, 1, &dev_id, "", NULL, NULL);
	if (status != CL_SUCCESS) {
		get_program_build_log(device_context.program, dev_id);
		throw std::runtime_error("OpenCL build failed");
	}

	device_context.k_init_ht = clCreateKernel(device_context.program, "kernel_init_ht", &status);
	if (status != CL_SUCCESS || !device_context.k_init_ht)
		throw std::runtime_error("clCreateKernel kernel_init_ht " + std::to_string(status));

	for (int round = 0; round < PARAM_K; round++) {
		char name[128];
		snprintf(name, sizeof(name), "kernel_round%d", round);
		device_context.k_rounds[round] = clCreateKernel(device_context.program, name, &status);
		if (status != CL_SUCCESS || !device_context.k_rounds[round])
			throw std::runtime_error("clCreateKernel kernel_round" + std::to_string(round) + " " + std::to_string(status));
	}

	device_context.k_sols = clCreateKernel(device_context.program, "kernel_sols", &status);
	if (status != CL_SUCCESS || !device_context.k_sols)
		throw std::runtime_error("clCreateKernel kernel_sols " + std::to_string(status));


	// initialize cl buffers 
	// FIXME didnt check this over
	device_context.buf_ht[0] = errClCreateBuffer(device_context.clcontext, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	device_context.buf_ht[1] = errClCreateBuffer(device_context.clcontext, CL_MEM_READ_WRITE, HT_SIZE, NULL);
	device_context.buf_sols = errClCreateBuffer(device_context.clcontext, CL_MEM_READ_WRITE, sizeof(sols_t), NULL);
	device_context.rowCounters[0] = errClCreateBuffer(device_context.clcontext, CL_MEM_READ_WRITE, NR_ROWS, NULL);
	device_context.rowCounters[0] = errClCreateBuffer(device_context.clcontext, CL_MEM_READ_WRITE, NR_ROWS, NULL);

}

// close command queues, etc
void opencl_solver::stop(opencl_solver& device_context) {
	cl_int status = 0;
	status |= clReleaseKernel(device_context.k_init_ht);
	for (int round = 0; round < PARAM_K; round++) {
		status |= clReleaseKernel(device_context.k_rounds[round]);
	}
	
	status |= clReleaseKernel(device_context.k_sols);
	status |= clReleaseProgram(device_context.program);
	status |= clReleaseCommandQueue(device_context.command_queue);
	status |= clReleaseContext(device_context.clcontext);
	
	if (status)
		throw std::runtime_error("Cleaning OpenCL resources failed");
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

	throw std::runtime_error("opencl_solver::solve()");

// TODO complete solve() implementation with ZCASH->AION adjustments
	ocl_blake2b::blake2b_state_t blake;
	cl_mem buf_blake_state;
	size_t global_work_size = device_context.numwg * device_context.wgsize;
	size_t local_work_size = device_context.wgsize;
	// FIXME
	uint32_t sol_found = 0;
	uint64_t* nonce_ptr;
	std::cout << tequihash_header_len<< " " << ZCASH_BLOCK_HEADER_LEN <<std::endl;
	std::cout << nonce_len << " " << std::endl;
	assert(tequihash_header_len == ZCASH_BLOCK_HEADER_LEN);
	
	// FIXME DEBUG
	// print nonce number 	
	
	ocl_blake2b::blake2b_init(&blake, ZCASH_HASH_LEN);
	ocl_blake2b::blake2b_update(&blake, reinterpret_cast<const uint8_t*>(tequihash_header), 128, 0);
	buf_blake_state = errClCreateBuffer(device_context.clcontext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(blake.h), &blake.h);

	for (int round = 0; round < PARAM_K; round ++) {
		initializeHashTable(device_context.command_queue, device_context.k_init_ht, device_context.buf_ht[round%2], device_context.rowCounters[round%2]);
		
		if (!round) {
			errClSetKernelArg(device_context.k_rounds[round], 0, &buf_blake_state);
			errClSetKernelArg(device_context.k_rounds[round], 1, &device_context.buf_ht[round % 2]);
			errClSetKernelArg(device_context.k_rounds[round], 2, &device_context.rowCounters[round % 2]);
			global_work_size = selectWorkSize();
		} else {
			errClSetKernelArg(device_context.k_rounds[round], 0, &device_context.buf_ht[(round - 1) % 2]);
			errClSetKernelArg(device_context.k_rounds[round], 1, &device_context.buf_ht[round % 2]);
			errClSetKernelArg(device_context.k_rounds[round], 2, &device_context.rowCounters[(round - 1) % 2]);
			errClSetKernelArg(device_context.k_rounds[round], 3, &device_context.rowCounters[round % 2]);
			global_work_size = NR_ROWS;
		}
		
		errClSetKernelArg(device_context.k_rounds[round], round == 0 ? 3 : 4, &device_context.buf_dbg);
		if (round == PARAM_K - 1) {
			errClSetKernelArg(device_context.k_rounds[round], 5, &device_context.buf_sols);
		errClEnqueueNDRangeKernel(device_context.command_queue, device_context.k_rounds[round], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

		// DEBUG examine_ht()
		}
	}
	errClSetKernelArg(device_context.k_sols, 0, &device_context.buf_ht[0]);
	errClSetKernelArg(device_context.k_sols, 1, &device_context.buf_ht[1]);
	errClSetKernelArg(device_context.k_sols, 2, &device_context.buf_sols);
	errClSetKernelArg(device_context.k_sols, 3, &device_context.rowCounters[0]);
	errClSetKernelArg(device_context.k_sols, 4, &device_context.rowCounters[1]);
	global_work_size = NR_ROWS;
	errClEnqueueNDRangeKernel(device_context.command_queue, device_context.k_sols, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);

}

