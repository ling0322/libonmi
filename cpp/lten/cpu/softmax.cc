// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "lten/cpu/softmax.h"

#include <cmath>

#include "lten/cpu/accessor.h"
#include "lten/cpu/tensor.h"
#include "lten/mp.h"

namespace lten {
namespace op {
namespace cpu {

template<typename T>
Tensor softmaxKernel(Tensor A) {
  Tensor C = tensorLike(A);
  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  MP::parallelFor({vA.getLength()}, [&vA, &vC](MP::Context ctx) {
    TensorAccessor<const T, 1> a = vA.getTensor(ctx.getBlockIdx());
    TensorAccessor<T, 1> c = vC.getTensor(ctx.getBlockIdx());

    std::vector<float> m(a.getShape(0) + 1);
    std::vector<float> d(a.getShape(0) + 1);
    m[0] = -1e10;
    d[0] = 0;
    for (int i = 0; i < a.getShape(0); i++) {
      T x = a[i];
      m[i + 1] = fmaxf(m[i], x);
      d[i + 1] = d[i] * expf(m[i] - m[i + 1]) + expf(x - m[i + 1]);
    }
    for (int i = 0; i < a.getShape(0); i++) {
      float x = a[i];
      c[i] = static_cast<T>(expf(x - m[a.getShape(0)]) / d[a.getShape(0)]);
    }
  });

  return C;
}

Tensor softmax(Tensor A) {
  if (A.getDType() == DType::kFloat) return softmaxKernel<float>(A);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16) return softmaxKernel<Float16>(A);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace lten
