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

#pragma once

#include <sstream>

#define LOG(severity)                                             \
  if (lut::internal::gLogLevel > lut::LogSeverity::k##severity) { \
  } else                                                          \
    lut::internal::LogWrapperk##severity(__FILE__, __LINE__)
#define NOT_IMPL()                   \
  {                                  \
    LOG(FATAL) << "not implemented"; \
    abort();                         \
  }

// CHECK macro conflicts with catch2
#define CHECK(cond) \
  if (cond) {       \
  } else            \
    LOG(FATAL).DefaultMessage("Check " #cond " failed.")

namespace lut {

enum class LogSeverity { kDEBUG = 0, kINFO = 1, kWARN = 2, kERROR = 4, kFATAL = 3 };

void setLogLevel(LogSeverity level);

}  // namespace lut

#include "lutil/internal/log.h"
