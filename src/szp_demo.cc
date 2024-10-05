#include "szp_demo.hh"

#include <type_traits>

#include "cuSZp_utility.h"
#include "szp_driver.hh"

namespace {

template <typename T>
T *read_data(std::string const fname, size_t *len)
{
  int status;
  char *fname2 = const_cast<char *>(fname.c_str());

  if constexpr (std::is_same_v<T, float>)
    return readFloatData_Yafan(fname2, len, &status);
  else if constexpr (std::is_same_v<T, double>)
    return readDoubleData_Yafan(fname2, len, &status);
  else
    return 0;
}

}  // namespace

void szp::compressor_roundtrip_float(
    std::string fname, int const x, int const y, int const z, double eb,
    bool use_rel)
{
  using T = float;
  szp::Compressor<T> cor(x, y, z);

  size_t len = x * y * z;
  double range;

  size_t _fake_len = 0;

  cor.input_hptr() = read_data<T>(fname, &len);
  cudaMalloc(&cor.input_dptr(), sizeof(T) * len);
  cudaMemset(cor.input_dptr(), 0, sizeof(T) * len);

  if (use_rel) {
    Compressor<T>::profile_data_range(cor.input_hptr(), len, range);
    eb *= range;
  }

  cudaMemcpy(
      cor.input_dptr(), cor.input_hptr(), sizeof(T) * len,
      cudaMemcpyHostToDevice);

  unsigned char *d_compressed_internal{nullptr};
  size_t compressed_size;

  cudaStream_t stream;  // external to compressor
  cudaStreamCreate(&stream);

  //// data is ready
  cor.compress(
      cor.input_dptr(), eb, &d_compressed_internal, &compressed_size, stream);

  printf("comp_metric::CR\t%f\n", 1.0 * len * sizeof(T) / compressed_size);

  //// mimick copying out compressed data after compression
  unsigned char *d_compressed_dump;

  cudaMalloc(&d_compressed_dump, compressed_size * sizeof(unsigned char));
  cudaMemcpy(
      d_compressed_dump, d_compressed_internal,
      sizeof(unsigned char) * compressed_size, cudaMemcpyDeviceToDevice);

  ////  mimick allocating for decompressed data before decompression
  T *d_decompressed;
  T *h_decompressed;
  cudaMalloc(&d_decompressed, sizeof(T) * len);
  cudaMallocHost(&h_decompressed, sizeof(T) * len);

  //// decompress using external saved archive
  cor.decompress(
      d_compressed_dump, compressed_size, eb, d_decompressed, stream);

  cudaMemcpy(
      h_decompressed, d_decompressed, sizeof(T) * len, cudaMemcpyDeviceToHost);

  //// check correctness
  int not_bound = 0;
  for (size_t i = 0; i < len; i += 1)
    if (abs(cor.input_hptr()[i] - h_decompressed[i]) > eb * 1.01) not_bound++;

  if (!not_bound)
    printf("\033[0;32mPass error check!\033[0m\n");
  else
    printf("\033[0;31mFail error check!\033[0m\n");

  //// clear up
  cudaFree(cor.input_dptr());
  cudaFree(d_compressed_dump);
  cudaFree(d_decompressed);
  cudaFreeHost(h_decompressed);

  cudaStreamDestroy(stream);
  delete[] cor.input_hptr();
}