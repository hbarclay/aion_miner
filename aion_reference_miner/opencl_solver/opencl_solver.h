
struct eq_opencl_context;

struct opencl_solver {
	int threadsperwv;
	int wv;
	int platform_id;
	int device_id;
	eq_opencl_context* context;	

	opencl_solver(int plat_id, int dev_id);
	std::string getdevinfo();

	static void getinfo(int plat_id, int dev_id, std::string& gpu_name, int& cu_count, std::string& version);

	static void start(opencl_solver& device_context);
	//static void start();

	static void stop(opencl_solver& device_context);
	//static void stop();

	static void solve(const char *tequihash_header,
			unsigned int tequihash_header_len, const char* nonce,
			unsigned int nonce_len, std::function<bool()> cancelf,
			std::function<
					void(const std::vector<uint32_t>&, size_t,
							const unsigned char*)> solutionf,
			std::function<void(void)> hashdonef,
			opencl_solver& device_context);

	std::string getname() {
		return "opencl_solver";
	}

};

