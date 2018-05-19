#pragma once

#include <AvailableSolvers.h>

class MinerFactory {
public:
	MinerFactory() {
	}

	~MinerFactory();

	std::vector<ISolver *> GenerateSolvers(int cpu_threads, int cuda_count,
			int* cuda_en, int* cuda_b, int* cuda_t, int opencl_count, int* opencl_plat_en, int* opencl_dev_en);
	void ClearAllSolvers();

private:
	std::vector<ISolver *> _solvers;

	ISolver * GenCPUSolver(int use_opt);
	ISolver * GenCUDASolver(int dev_id, int blocks, int threadsperblock);
	ISolver * GenOpenCLSolver(int plat_id, int dev_id, int threadsperwf, int wfpersimd);
};

