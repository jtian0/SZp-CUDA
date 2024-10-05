#ifndef ADCB188C_E12B_4F28_A380_879DB93F814F
#define ADCB188C_E12B_4F28_A380_879DB93F814F

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SZP_SUCCESS,
  SZP_GENERAL_GPU_FAILURE,  // translate from all cuda_errors
  SZP_NOT_IMPLEMENTED,
  SZP_WRONG_TYPE,
} szp_error_status;

typedef szp_error_status szperror;

#ifdef __cplusplus
}
#endif

#define NON_INTRUSIVE_MOD_2410 1

namespace szp::cuhip::utils {

template <typename T>
size_t get_bsize();

template <typename T>
size_t get_bsize();

template <typename T>
size_t get_comp_off_size(size_t len);

}  // namespace szp::cuhip::utils

namespace szp::cuhip {

template <typename T>
szperror GPU_compress_singleton(
    T *d_in, size_t const len,                   // input
    unsigned char *d_cmpBytes, size_t *cmpSize,  // output
    unsigned int *uni_cmpOffset, int *d_flag,    // buffer
    float const eb, void *stream);

template <typename T>
szperror GPU_decompress_singleton(
    T *d_out, size_t const len,  // output
    unsigned char *d_cmpBytes,   // input
    size_t const cmpSize,        //
    unsigned int *d_cmpOffset,   // buffer
    int *d_flag,                 //
    float const eb, void *stream);

}  // namespace szp::cuhip

#endif /* ADCB188C_E12B_4F28_A380_879DB93F814F */
