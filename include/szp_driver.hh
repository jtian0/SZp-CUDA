#ifndef A87B3D52_DD91_4521_8ACB_35156A653D6A
#define A87B3D52_DD91_4521_8ACB_35156A653D6A

#include <cuda_runtime.h>

#include <chrono>

#include "szp_module.hh"

#define CHECK_CUDA2(err) (szp::check_cuda_error(err, __FILE__, __LINE__))

namespace szp {

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::duration<double>;

template <typename T>
class internal_membuf {
 public:
  size_t const len;
  size_t cmpOffSize;

  // ref to external
  T *d_input;
  T *h_input;

  // internal
  unsigned int *uni_cmpOffset;
  int *d_flag;
  unsigned char *d_cmpBytes;

  internal_membuf(size_t _len);
  ~internal_membuf();
};

template <typename T>
class Compressor {
 private:
  int const x, y, z;
  size_t len;

  szp::internal_membuf<T> *buf;

 public:
  szp::time_t comp_start, comp_end, decomp_start, decomp_end;

  Compressor(int const _x, int const _y, int const _z) : x(_x), y(_y), z(_z)
  {
    len = x * y * z;
    buf = new szp::internal_membuf<T>(x * y * z);
  };
  ~Compressor() { delete buf; };

  static void profile_data_range(
      float *h_input, size_t const len, double &range);

  void compress(
      T *d_in, double eb, unsigned char **pd_archive, size_t *archive_size,
      void *stream);

  void decompress(
      unsigned char *d_archive, size_t const archive_size, double const eb,
      T *d_out, void *stream);

  // getter (testing purposes)
  T *&input_hptr() const { return buf->h_input; };
  T *&input_dptr() const { return buf->d_input; };
  //   size_t comp_size() const { return 1;
  szp::internal_membuf<T> *const &membuf() const { return buf; };
};

}  // namespace szp

#endif /* A87B3D52_DD91_4521_8ACB_35156A653D6A */
