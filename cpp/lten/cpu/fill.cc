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

#include "lten/cpu/fill.h"

#include "lten/cpu/accessor.h"
#include "lten/cpu/common.h"
#include "lten/cpu/tensor.h"
#include "lten/mp.h"
#include "lten/tensor.h"

namespace lten {
namespace op {
namespace cpu {

template<typename T>
void fillKernel(Tensor A, float value) {
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(A);
  MP::parallelFor(vC.getLength(), [&vC, value](MP::Context ctx) {
    TensorAccessor<T, 1> c = vC.getTensor(ctx.getBlockIdx());

    for (int i = 0; i < c.getShape(0); ++i) {
      c[i] = value;
    }
  });
}

void fill(Tensor src, float value) {
  if (src.getDType() == DType::kFloat) {
    if (src.getNumEl() == 1) {
      *src.getData<float>() = value;
    } else {
      fillKernel<float>(src, value);
    }
    return;
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  if (src.getDType() == DType::kFloat16) {
    if (src.getNumEl() == 1) {
      *src.getData<Float16>() = value;
    } else {
      fillKernel<Float16>(src, value);
    }
    return;
  }
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace lten
