// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "lten/cpu/kernel/abstract.h"

// UInt4x2 -> UInt8 SIMD
// read 32 int4 (16 bytes), convert to 32 int8 and store to xi8x32.
// Here is the steps of converting int4 to int8:
//
// Input:
// High ----- Low
// +---+---+
// | B | A | <- packed 2 uint4 values A and B  into a byte
// +---+---+
//
// u8 -> i16 (1)
// +---+---+---+---+
// | 0 | 0 | B | A |
// +---+---+---+---+
//
// i16 SHIFT-LEFT 4 (2)
// +---+---+---+---+
// | 0 | B | A | 0 |
// +---+---+---+---+
//
// i16 (1) OR (2)
// +---+---+---+---+
// | 0 | B | X | A |
// +---+---+---+---+
//
// As 2 int8 (little-endian)
// +---+---+  +---+---+
// | 0 | A |  | X | B |
// +---+---+  +---+---+
//
// AND 0xf
// +---+---+  +---+---+
// | 0 | A |  | 0 | B |
// +---+---+  +---+---+

namespace lten {
namespace op {
namespace cpu {
namespace kernel {

#if LIBLLM_KERNEL_MSVC
inline float libllm_cvtsh_ss(uint16_t sh) {
  __m128i vh = _mm_set1_epi16(sh);
  __m128 vs = _mm_cvtph_ps(vh);
  return _mm_cvtss_f32(vs);
}
#endif

LIBLLM_KERNEL_FORCE_INLINE float hsum(__m256 ymm) {
  __m128 x = _mm256_castps256_ps128(ymm);
  x = _mm_add_ps(x, _mm256_extractf128_ps(ymm, 1));
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ps(x, _mm_movehdup_ps(x));
  return _mm_cvtss_f32(x);
}

void sgemm6x16Avx2Kernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR (6 x 2 ymmX)
  __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  __m256 a00, b00, b01;

  float *pc = c;
  c00 = _mm256_loadu_ps(pc);
  c01 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c10 = _mm256_loadu_ps(pc);
  c11 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c20 = _mm256_loadu_ps(pc);
  c21 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c30 = _mm256_loadu_ps(pc);
  c31 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c40 = _mm256_loadu_ps(pc);
  c41 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c50 = _mm256_loadu_ps(pc);
  c51 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  const float *pa = a;
  const float *pb = b;
  for (int k = 0; k < kc; ++k) {
    b00 = _mm256_loadu_ps(pb);
    b01 = _mm256_loadu_ps(pb + 8);
    a00 = _mm256_broadcast_ss(pa);

    c00 = _mm256_fmadd_ps(a00, b00, c00);
    c01 = _mm256_fmadd_ps(a00, b01, c01);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c10 = _mm256_fmadd_ps(a00, b00, c10);
    c11 = _mm256_fmadd_ps(a00, b01, c11);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c20 = _mm256_fmadd_ps(a00, b00, c20);
    c21 = _mm256_fmadd_ps(a00, b01, c21);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c30 = _mm256_fmadd_ps(a00, b00, c30);
    c31 = _mm256_fmadd_ps(a00, b01, c31);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c40 = _mm256_fmadd_ps(a00, b00, c40);
    c41 = _mm256_fmadd_ps(a00, b01, c41);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c50 = _mm256_fmadd_ps(a00, b00, c50);
    c51 = _mm256_fmadd_ps(a00, b01, c51);
    pa += 1;

    pb += 16;
  }

  pc = c;
  _mm256_storeu_ps(pc, c00);
  _mm256_storeu_ps(pc + 8, c01);
  pc += rs_c;

  _mm256_storeu_ps(pc, c10);
  _mm256_storeu_ps(pc + 8, c11);
  pc += rs_c;

  _mm256_storeu_ps(pc, c20);
  _mm256_storeu_ps(pc + 8, c21);
  pc += rs_c;

  _mm256_storeu_ps(pc, c30);
  _mm256_storeu_ps(pc + 8, c31);
  pc += rs_c;

  _mm256_storeu_ps(pc, c40);
  _mm256_storeu_ps(pc + 8, c41);
  pc += rs_c;

  _mm256_storeu_ps(pc, c50);
  _mm256_storeu_ps(pc + 8, c51);
  pc += rs_c;
}

void saxpyAvx2Kernel(int64_t n, float a, const float *x, float *y) {
  __m256 a00 = _mm256_broadcast_ss(&a);
  __m256 x00, y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);

    y00 = _mm256_fmadd_ps(a00, x00, y00);
    _mm256_storeu_ps(py, y00);

    px += 8;
    py += 8;
  }

  for (int i = 0; i < nr; ++i) {
    *py++ += a * *px++;
  }
}

float sdotAvx2Kernel(int64_t n, const float *x, const float *y) {
  __m256 x00, y00, a00;

  a00 = _mm256_setzero_ps();

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  const float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);
    a00 = _mm256_fmadd_ps(x00, y00, a00);

    px += 8;
    py += 8;
  }

  // unroll a00
  float sum = hsum(a00);
  for (int i = 0; i < nr; ++i) {
    sum += *px++ * *py++;
  }

  return sum;
}

LIBLLM_KERNEL_FORCE_INLINE __m256i loadNibble32ToByte32(const void *nibbleAddr) {
  __m256i vbyte = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)nibbleAddr));
  vbyte = _mm256_or_si256(_mm256_slli_epi16(vbyte, 4), vbyte);
  vbyte = _mm256_and_si256(vbyte, _mm256_set1_epi8(0xf));
  return vbyte;
}

LIBLLM_KERNEL_FORCE_INLINE float half2float(Float16 half) {
#if LIBLLM_KERNEL_MSVC
  return libllm_cvtsh_ss(*reinterpret_cast<uint16_t *>(&half));
#else
  return _cvtsh_ss(*reinterpret_cast<uint16_t *>(&half));
#endif
}

LIBLLM_KERNEL_FORCE_INLINE __m256 extractFloat8FromByte32Block0(__m256i src) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(src, 0)));
}

LIBLLM_KERNEL_FORCE_INLINE __m256 extractFloat8FromByte32Block1(__m256i src) {
  return _mm256_cvtepi32_ps(
      _mm256_cvtepi8_epi32(_mm_srli_si128(_mm256_extracti128_si256(src, 0), 8)));
}

LIBLLM_KERNEL_FORCE_INLINE __m256 extractFloat8FromByte32Block2(__m256i src) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256(src, 1)));
}

LIBLLM_KERNEL_FORCE_INLINE __m256 extractFloat8FromByte32Block3(__m256i src) {
  return _mm256_cvtepi32_ps(
      _mm256_cvtepi8_epi32(_mm_srli_si128(_mm256_extracti128_si256(src, 1), 8)));
}

float sqdotAvx2Kernel(int64_t n, const float *x, const QInt4x32 *y, int64_t offsetY) {
  __m256 vx, vy, vsum, vscale, vzero;
  __m256i vbytey;

  vsum = _mm256_setzero_ps();
  int64_t groupIdx = offsetY / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetY % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  const float *px = x;
  const QInt4x32 *py = y + groupIdx;
  for (int i = groupIdx; i < groupIdx + nb; ++i) {
    vscale = _mm256_set1_ps(half2float(py->scale));
    vzero = _mm256_set1_ps(-half2float(py->zero));

    // block 0
    vbytey = loadNibble32ToByte32(py->data);

    // block 0:0
    vy = _mm256_fmadd_ps(extractFloat8FromByte32Block0(vbytey), vscale, vzero);
    vx = _mm256_loadu_ps(px);
    vsum = _mm256_fmadd_ps(vx, vy, vsum);
    px += 8;

    // block 0:1
    vy = _mm256_fmadd_ps(extractFloat8FromByte32Block1(vbytey), vscale, vzero);
    vx = _mm256_loadu_ps(px);
    vsum = _mm256_fmadd_ps(vx, vy, vsum);
    px += 8;

    // block 0:2
    vy = _mm256_fmadd_ps(extractFloat8FromByte32Block2(vbytey), vscale, vzero);
    vx = _mm256_loadu_ps(px);
    vsum = _mm256_fmadd_ps(vx, vy, vsum);
    px += 8;

    // block 0:3
    vy = _mm256_fmadd_ps(extractFloat8FromByte32Block3(vbytey), vscale, vzero);
    vx = _mm256_loadu_ps(px);
    vsum = _mm256_fmadd_ps(vx, vy, vsum);
    px += 8;

    ++py;
  }

  return hsum(vsum);
}

void qscvtAvx2Kernel(int n, const QInt4x32 *x, int64_t offsetX, float *y) {
  __m256 vx, vscale, vzero;
  __m256i vbytex;

  int64_t groupIdx = offsetX / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetX % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  const QInt4x32 *px = x + groupIdx;
  float *py = y;

  for (int64_t i = groupIdx; i < groupIdx + nb; ++i) {
    vscale = _mm256_set1_ps(half2float(px->scale));
    vzero = _mm256_set1_ps(-half2float(px->zero));

    // block 0
    vbytex = loadNibble32ToByte32(px->data);
    vx = _mm256_fmadd_ps(extractFloat8FromByte32Block0(vbytex), vscale, vzero);
    _mm256_storeu_ps(py, vx);
    py += 8;
    vx = _mm256_fmadd_ps(extractFloat8FromByte32Block1(vbytex), vscale, vzero);
    _mm256_storeu_ps(py, vx);
    py += 8;
    vx = _mm256_fmadd_ps(extractFloat8FromByte32Block2(vbytex), vscale, vzero);
    _mm256_storeu_ps(py, vx);
    py += 8;
    vx = _mm256_fmadd_ps(extractFloat8FromByte32Block3(vbytex), vscale, vzero);
    _mm256_storeu_ps(py, vx);
    py += 8;

    ++px;
  }
}

void hscvtAvx2Kernel(int64_t n, const Float16 *x, float *y) {
  int nb = n / 8;
  for (int i = 0; i < nb; ++i) {
    __m128i x0 = _mm_loadu_si128((const __m128i *)x);
    __m256 y0 = _mm256_cvtph_ps(x0);
    _mm256_storeu_ps(y, y0);

    x += 8;
    y += 8;
  }

  int nr = n % 8;
  if (nr == 0) return;

  Float16 xr[8];
  float yr[8];
  for (int i = 0; i < nr; ++i) {
    xr[i] = x[i];
  }
  __m128i x0 = _mm_loadu_si128((const __m128i *)xr);
  __m256 y0 = _mm256_cvtph_ps(x0);
  _mm256_storeu_ps(yr, y0);
  for (int i = 0; i < nr; ++i) {
    y[i] = yr[i];
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace lten
