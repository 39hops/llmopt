/* Canonical Huffman codec for 8-bit symbols, block-structured with byte
 * aligned block boundaries (the DFloat11-style exponent coder of
 * scratch/bf16_lossless_bench.py). Codes are MSB-first, canonical
 * (assigned from the code lengths in (length, symbol) order), at most
 * 56 bits long. Compiled at run time with clang into the job's tmp
 * directory and loaded with ctypes; never installed.
 *
 * encode: writes blocks of `block` symbols; block b's bytes start at
 * offsets[b] (offsets[nblocks] = total bytes). Returns total bytes or -1.
 * decode: reverses it; returns 0, or -1 on a malformed stream (a code
 * that matches no symbol, or a block that runs past its byte span). */
#include <stdint.h>
#include <string.h>

static int build_codes(const uint8_t *lens, uint64_t *codes, int *maxlen) {
    int count[64] = {0};
    int ml = 0;
    for (int s = 0; s < 256; s++) { if (lens[s] > 56) return -1; count[lens[s]]++; if (lens[s] > ml) ml = lens[s]; }
    count[0] = 0;
    uint64_t code = 0, next[64];
    for (int l = 1; l <= 56; l++) { code = (code + count[l - 1]) << 1; next[l] = code; }
    for (int s = 0; s < 256; s++) codes[s] = lens[s] ? next[lens[s]]++ : 0;
    *maxlen = ml;
    return 0;
}

int64_t huff_encode(const uint8_t *sym, int64_t n, const uint8_t *lens, int64_t block,
                    uint8_t *out, int64_t out_cap, int64_t *offsets) {
    uint64_t codes[256]; int ml;
    if (build_codes(lens, codes, &ml)) return -1;
    int64_t nblocks = (n + block - 1) / block, pos = 0;
    for (int64_t b = 0; b < nblocks; b++) {
        offsets[b] = pos;
        int64_t s0 = b * block, s1 = s0 + block < n ? s0 + block : n;
        uint64_t buf = 0; int nb = 0;
        for (int64_t i = s0; i < s1; i++) {
            int l = lens[sym[i]];
            if (!l) return -1;
            buf = (buf << l) | codes[sym[i]]; nb += l;
            while (nb >= 8) { if (pos >= out_cap) return -1; out[pos++] = (uint8_t)(buf >> (nb - 8)); nb -= 8; }
        }
        if (nb) { if (pos >= out_cap) return -1; out[pos++] = (uint8_t)(buf << (8 - nb)); }
    }
    offsets[nblocks] = pos;
    return pos;
}

int huff_decode(const uint8_t *in, const int64_t *offsets, int64_t n, const uint8_t *lens, int64_t block,
                uint8_t *out) {
    int count[64] = {0}; uint8_t order[256]; int ml = 0;
    for (int s = 0; s < 256; s++) { if (lens[s] > 56) return -1; count[lens[s]]++; if (lens[s] > ml) ml = lens[s]; }
    count[0] = 0;
    /* symbols sorted by (length, symbol): canonical order */
    int k = 0;
    for (int l = 1; l <= ml; l++) for (int s = 0; s < 256; s++) if (lens[s] == l) order[k++] = (uint8_t)s;
    int64_t first_code[64], first_idx[64];
    { uint64_t code = 0; int64_t idx = 0;
      for (int l = 1; l <= ml; l++) { code = (code + count[l - 1]) << 1; first_code[l] = (int64_t)code; first_idx[l] = idx; idx += count[l]; } }
    int64_t nblocks = (n + block - 1) / block;
    for (int64_t b = 0; b < nblocks; b++) {
        int64_t s0 = b * block, s1 = s0 + block < n ? s0 + block : n;
        const uint8_t *p = in + offsets[b], *pe = in + offsets[b + 1];
        uint64_t buf = 0; int nb = 0;
        for (int64_t i = s0; i < s1; i++) {
            int64_t code = 0; int l = 0; int found = 0;
            while (l < ml) {
                if (nb == 0) { if (p >= pe) return -1; buf = *p++; nb = 8; }
                code = (code << 1) | ((buf >> (nb - 1)) & 1u); nb--; l++;
                if (count[l] && code >= first_code[l] && code < first_code[l] + count[l]) {
                    out[i] = order[first_idx[l] + (code - first_code[l])]; found = 1; break;
                }
            }
            if (!found) return -1;
        }
    }
    return 0;
}
