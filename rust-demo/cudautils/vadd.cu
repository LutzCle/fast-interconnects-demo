// Add a scalar to the vector
//
// IMPORTANT: Prevent symbol mangling by setting: extern "C"
extern "C"
__global__ void vadd(int *const v, int const a, size_t const len) {
	const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int gsize = gridDim.x * blockDim.x;

	for (size_t i = gid; i < len; i += gsize) {
		v[i] += a;
	}
}
