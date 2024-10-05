#include "szp_driver.hh"

#include <algorithm>
#include <cstdio>
#include <type_traits>

namespace szp {

void check_cuda_error(cudaError_t status, const char *file, int line)
{
  if (cudaSuccess != status) {
    printf("\n");
    printf(
        "CUDA API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n",  //
        file, line, cudaGetErrorString(status), status);
    // exit(EXIT_FAILURE);
  }
}

template <typename T>
internal_membuf<T>::internal_membuf(size_t _len) : len(_len)
{
  auto cmpOffSize = szp::cuhip::utils::get_comp_off_size<T>(len);

  CHECK_CUDA2(cudaMalloc(&d_cmpBytes, sizeof(T) * len));
  CHECK_CUDA2(
      cudaMallocManaged(&uni_cmpOffset, sizeof(unsigned int) * cmpOffSize));
  CHECK_CUDA2(cudaMemset(uni_cmpOffset, 0, sizeof(unsigned int) * cmpOffSize));
  CHECK_CUDA2(cudaMalloc(&d_flag, sizeof(int) * cmpOffSize));
  CHECK_CUDA2(cudaMemset(d_flag, 0, sizeof(int) * cmpOffSize));
};

template <typename T>
internal_membuf<T>::~internal_membuf()
{
  cudaFree(d_cmpBytes);
  cudaFree(uni_cmpOffset);
  cudaFree(d_flag);
}

template <typename T>
void Compressor<T>::profile_data_range(
    float *h_input, size_t const len, double &range)
{
  range = *std::max_element(h_input, h_input + len) -
          *std::min_element(h_input, h_input + len);
}

template <typename T>
void Compressor<T>::compress(
    T *d_in, double eb, unsigned char **pd_archive, size_t *archive_size,
    void *stream)
{
  comp_start = std::chrono::system_clock::now();

  // The callee contains synchronization by stream
  szp::cuhip::GPU_compress_singleton<T>(
      d_in, len, buf->d_cmpBytes, archive_size, buf->uni_cmpOffset,
      buf->d_flag, eb, stream);

  comp_end = std::chrono::system_clock::now();
}

template <typename T>
void Compressor<T>::decompress(
    unsigned char *d_archive, size_t const archive_size, double const eb,
    T *d_out, void *stream)
{
  decomp_start = std::chrono::system_clock::now();

  // The callee contains synchronization by stream
  szp::cuhip::GPU_decompress_singleton<T>(
      d_out, len, d_archive, archive_size, buf->uni_cmpOffset, buf->d_flag, eb,
      stream);

  decomp_end = std::chrono::system_clock::now();
}

}  // namespace szp

template class szp::Compressor<float>;
template class szp::Compressor<double>;