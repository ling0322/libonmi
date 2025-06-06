/*
 * Base64 encoding/decoding (RFC1341)
 * Copyright (c) 2005, Jouni Malinen <j@w1.fi>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * Alternatively, this software may be distributed under the terms of BSD
 * license.
 *
 * See README and COPYING for more details.
 */

// original file:
// https://android.googlesource.com/platform/external/wpa_supplicant/+/4d8c3c1ca334d1319decf3e2c5d2be0cf472e3f9/base64.c

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "lutil/span.h"

static const unsigned char
    base64_table[65] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

namespace lut {
namespace internal {

/**
 * base64_encode - Base64 encode
 * @src: Data to be encoded
 * @len: Length of the data to be encoded
 * @out_len: Pointer to output length variable, or %NULL if not used
 * Returns: Allocated buffer of out_len bytes of encoded data,
 * or %NULL on failure
 *
 * Caller is responsible for freeing the returned buffer. Returned buffer is
 * nul terminated to make it easier to use as a C string. The nul terminator is
 * not included in out_len.
 */
unsigned char *base64_encode(const unsigned char *src, size_t len, size_t *out_len) {
  unsigned char *out, *pos;
  const unsigned char *end, *in;
  size_t olen;
  int line_len;
  olen = len * 4 / 3 + 4; /* 3-byte blocks to 4-byte */
  olen += olen / 72;      /* line feeds */
  olen++;                 /* nul termination */
  out = (unsigned char *)malloc(olen);
  if (out == nullptr) return nullptr;
  end = src + len;
  in = src;
  pos = out;
  line_len = 0;
  while (end - in >= 3) {
    *pos++ = base64_table[in[0] >> 2];
    *pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
    *pos++ = base64_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
    *pos++ = base64_table[in[2] & 0x3f];
    in += 3;
    line_len += 4;
    if (line_len >= 72) {
      *pos++ = '\n';
      line_len = 0;
    }
  }
  if (end - in) {
    *pos++ = base64_table[in[0] >> 2];
    if (end - in == 1) {
      *pos++ = base64_table[(in[0] & 0x03) << 4];
      *pos++ = '=';
    } else {
      *pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
      *pos++ = base64_table[(in[1] & 0x0f) << 2];
    }
    *pos++ = '=';
    line_len += 4;
  }
  if (line_len) *pos++ = '\n';
  *pos = '\0';
  if (out_len) *out_len = pos - out;
  return out;
}
/**
 * base64_decode - Base64 decode
 * @src: Data to be decoded
 * @len: Length of the data to be decoded
 * @out_len: Pointer to output length variable
 * Returns: Allocated buffer of out_len bytes of decoded data,
 * or %NULL on failure
 *
 * Caller is responsible for freeing the returned buffer.
 */
unsigned char *base64_decode(const unsigned char *src, size_t len, size_t *out_len) {
  unsigned char dtable[256], *out, *pos, in[4], block[4], tmp;
  size_t i, count, olen;
  memset(dtable, 0x80, 256);
  for (i = 0; i < sizeof(base64_table) - 1; i++) dtable[base64_table[i]] = (unsigned char)i;
  dtable['='] = 0;
  count = 0;
  for (i = 0; i < len; i++) {
    if (dtable[src[i]] != 0x80) count++;
  }
  if (count == 0 || count % 4) return nullptr;
  olen = count / 4 * 3;
  pos = out = (unsigned char *)malloc(olen);
  if (out == nullptr) return nullptr;
  count = 0;
  for (i = 0; i < len; i++) {
    tmp = dtable[src[i]];
    if (tmp == 0x80) continue;
    in[count] = src[i];
    block[count] = tmp;
    count++;
    if (count == 4) {
      *pos++ = (block[0] << 2) | (block[1] >> 4);
      *pos++ = (block[1] << 4) | (block[2] >> 2);
      *pos++ = (block[2] << 6) | block[3];
      count = 0;
    }
  }
  if (pos > out) {
    if (in[2] == '=')
      pos -= 2;
    else if (in[3] == '=')
      pos--;
  }
  *out_len = pos - out;
  return out;
}

}  // namespace internal
}  // namespace lut

namespace lut {

std::vector<int8_t> decodeBase64(const std::string &base64String) {
  size_t outlen;
  int8_t *pdata = reinterpret_cast<int8_t *>(internal::base64_decode(
      reinterpret_cast<const unsigned char *>(base64String.c_str()),
      base64String.size(),
      &outlen));

  std::vector<int8_t> output(outlen);
  std::copy(pdata, pdata + outlen, output.begin());

  free(pdata);
  return output;
}

std::string encodeBase64(lut::Span<const int8_t> data) {
  size_t outlen;
  char *pstring = reinterpret_cast<char *>(internal::base64_encode(
      reinterpret_cast<const unsigned char *>(data.data()),
      data.size(),
      &outlen));

  std::string s(pstring, outlen);

  free(pstring);
  return s;
}

}  // namespace lut