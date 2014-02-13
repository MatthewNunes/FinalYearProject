#define CUDA_CHECK_RETURN(val)                      \
  {                                 \
    cudaError_t cuda_val = val;                     \
    if (cuda_val != cudaSuccess)                    \
      {                                 \
    fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(cuda_val), __LINE__, __FILE__); \
    exit(-1);                           \
      }                                 \
  }