#include <immintrin.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

inline float dot_product_avx512(const float* a, const float* b, int n) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    int i = 0;

    for (; i + 63 < n; i += 64) {
        __m512 a0 = _mm512_loadu_ps(&a[i]);
        __m512 b0 = _mm512_loadu_ps(&b[i]);
        sum0 = _mm512_fmadd_ps(a0, b0, sum0);

        __m512 a1 = _mm512_loadu_ps(&a[i + 16]);
        __m512 b1 = _mm512_loadu_ps(&b[i + 16]);
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);

        __m512 a2 = _mm512_loadu_ps(&a[i + 32]);
        __m512 b2 = _mm512_loadu_ps(&b[i + 32]);
        sum2 = _mm512_fmadd_ps(a2, b2, sum2);

        __m512 a3 = _mm512_loadu_ps(&a[i + 48]);
        __m512 b3 = _mm512_loadu_ps(&b[i + 48]);
        sum3 = _mm512_fmadd_ps(a3, b3, sum3);
    }

    for (; i + 15 < n; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(&a[i]);
        __m512 b_vec = _mm512_loadu_ps(&b[i]);
        sum0 = _mm512_fmadd_ps(a_vec, b_vec, sum0);
    }

    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);

    float sum = _mm512_reduce_add_ps(sum0);

    for (; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}
