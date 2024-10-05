#include "szp_module.hh"

namespace szp {

#include "cuSZp_f32.cu"
#include "cuSZp_f64.cu"

}  // namespace szp

#include "szp_driver.hh"

namespace szp::cuhip::utils {

static const int cmp_tblock_size_f32 = 32;
static const int dec_tblock_size_f32 = 32;
static const int cmp_chunk_f32 = 8192;
static const int dec_chunk_f32 = 8192;

static const int cmp_tblock_size_f64 = 32;
static const int dec_tblock_size_f64 = 32;
static const int cmp_chunk_f64 = 8192;
static const int dec_chunk_f64 = 8192;

template <typename T>
size_t get_bsize()
{
  if constexpr (std::is_same_v<T, float>)
    return cmp_tblock_size_f32;
  else
    return cmp_tblock_size_f64;
  return 0;
};

template <typename T>
size_t get_gsize(size_t len)
{
  if constexpr (std::is_same_v<T, float>)
    return (len + get_bsize<T>() * cmp_chunk_f32 - 1) /
           (get_bsize<T>() * cmp_chunk_f32);
  else
    return (len + get_bsize<T>() * cmp_chunk_f64 - 1) /
           (get_bsize<T>() * dec_chunk_f64);
  return 0;
}

template <typename T>
size_t get_comp_off_size(size_t len)
{
  return get_gsize<T>(len) + 1;
}

}  // namespace szp::cuhip::utils

namespace szp::cuhip {

template <typename T>
szperror GPU_compress_singleton(
    T *d_in, size_t const len,                   // input
    unsigned char *d_cmpBytes, size_t *cmpSize,  // output
    unsigned int *uni_cmpOffset, int *d_flag,    // buffer
    float const eb, void *stream)
{
  auto bsize = szp::cuhip::utils::get_bsize<T>();
  auto gsize = szp::cuhip::utils::get_gsize<T>(len);
  auto cmpOffSize = szp::cuhip::utils::get_comp_off_size<T>(len);

  dim3 blockSize(bsize);
  dim3 gridSize(gsize);

  if constexpr (std::is_same_v<T, float>) {
    KERNEL_CUHIP_szp_compress_singleton_f4                  //
        <<<gridSize, blockSize, 0, (cudaStream_t)stream>>>  //
        (d_in, d_cmpBytes, uni_cmpOffset, d_flag, eb, len);
  }
  else if constexpr (std::is_same_v<T, double>) {
    KERNEL_CUHIP_szp_decompress_singleton_f8                //
        <<<gridSize, blockSize, 0, (cudaStream_t)stream>>>  //
        (d_in, d_cmpBytes, uni_cmpOffset, d_flag, eb, len);
  }
  else {
    return SZP_WRONG_TYPE;
  }
  cudaStreamSynchronize((cudaStream_t)stream);

  *cmpSize = (size_t)uni_cmpOffset[cmpOffSize - 1] + (len + 31) / 32;

  return SZP_SUCCESS;
}

template <typename T>
szperror GPU_decompress_singleton(
    T *d_out, size_t const len,  // output
    unsigned char *d_cmpBytes,   // input
    size_t const cmpSize,        //
    unsigned int *d_cmpOffset,   // buffer
    int *d_flag,                 //
    float const eb, void *stream)
{
  auto bsize = szp::cuhip::utils::get_bsize<T>();
  auto gsize = szp::cuhip::utils::get_gsize<T>(len);
  auto cmpOffSize = szp::cuhip::utils::get_comp_off_size<T>(len);

  dim3 blockSize(bsize);
  dim3 gridSize(gsize);

  if constexpr (std::is_same_v<T, float>) {
    KERNEL_CUHIP_szp_decompress_singleton_f4                //
        <<<gridSize, blockSize, 0, (cudaStream_t)stream>>>  //
        (d_out, d_cmpBytes, d_cmpOffset, d_flag, eb, len);
  }
  else if constexpr (std::is_same_v<T, double>) {
    KERNEL_CUHIP_szp_decompress_singleton_f8                //
        <<<gridSize, blockSize, 0, (cudaStream_t)stream>>>  //
        (d_out, d_cmpBytes, d_cmpOffset, d_flag, eb, len);
  }
  else {
    return SZP_WRONG_TYPE;
  }
  cudaStreamSynchronize((cudaStream_t)stream);

  return SZP_SUCCESS;
}

}  // namespace szp::cuhip

template szperror szp::cuhip::GPU_compress_singleton<float>(
    float *, size_t const, unsigned char *, size_t *, unsigned int *, int *,
    float const, void *);

template szperror szp::cuhip::GPU_compress_singleton<double>(
    double *, size_t const, unsigned char *, size_t *, unsigned int *, int *,
    float const, void *);

template szperror szp::cuhip::GPU_decompress_singleton<float>(
    float *, size_t const, unsigned char *, size_t const, unsigned int *,
    int *, float const, void *);

template szperror szp::cuhip::GPU_decompress_singleton<double>(
    double *, size_t const, unsigned char *, size_t const, unsigned int *,
    int *, float const, void *);