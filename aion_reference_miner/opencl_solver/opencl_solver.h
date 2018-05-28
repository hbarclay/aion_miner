#include <string>
#include <functional>
#include <vector>
#include <ostream>

#include <CL/cl.h>
typedef unsigned char uchar;
#include "param.h"

struct eq_opencl_context;

struct opencl_solver {
 public:
	int numwg;
	int wgsize;
	int platform_id;
	int device_id;

	eq_opencl_context* context;	


	opencl_solver(int plat_id, int dev_id) : platform_id(plat_id), device_id(dev_id) {}
	std::string getdevinfo();


	static void print_opencl_info(std::ostream& os);
	static void getinfo(int plat_id, int dev_id, std::string& gpu_name, int& cu_count, std::string& version);

	static void start(opencl_solver& device_context);
	static void stop(opencl_solver& device_context);
	static void solve(const char *tequihash_header,
			unsigned int tequihash_header_len, const char* nonce,
			unsigned int nonce_len, std::function<bool()> cancelf,
			std::function<
					void(const std::vector<uint32_t>&, size_t,
							const unsigned char*)> solutionf,
			std::function<void(void)> hashdonef,
			opencl_solver& device_context);

	static std::string getname() {
		return "opencl_solver";
	}

 private:
	cl_context clcontext;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel k_init_ht, k_sols;
	cl_kernel k_rounds[PARAM_K];

	cl_mem buf_ht[2], buf_sols, buf_dbg, rowCounters[2];

};

