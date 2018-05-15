
#include <stdint.h>
#include <string.h>
#include "blake2b.h"

// TODO add value checks

#define PARAM_N 210
#define PARAM_K 9

#define BLAKE2B_ROUNDS 12
#define BLAKE2B_BLOCKSIZE 128


namespace {

static const uint64_t blake2b_IV[8] =
{
	0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
	0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
	0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
	0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// 10 rows, rows 11 and 12 are rows 1 and 2
static const uint8_t blake2b_SIGMA[12][16] =
{
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
	{7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
	{2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
	{13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
	{10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
	{14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

static inline uint64_t rotr64(uint64_t n, unsigned int c) {
	const unsigned int mask = (8*sizeof(n) - 1);
	c &= mask;
	return (n << c) | (n >> ((-c) & mask));
}

static inline void mix(uint64_t* Va, uint64_t* Vb, uint64_t* Vc, uint64_t* Vd, uint64_t x, uint64_t y) {
	*Va = *Va + *Vb + x;
	*Vd = rotr64(*Vd ^ *Va, 32);

	*Vc = (*Vc + *Vd);
	*Vb = rotr64(*Vb ^ *Vc, 24);

	*Va = *Va + *Vb + y;
	*Vd = rotr64(*Vd ^ *Va, 16);

	*Vc = *Vc + *Vd;
	*Vb = rotr64(*Vb ^ *Vc, 63);
}

}  // namespace

void ocl_blake2b::blake2b_init(blake2b_state_t* state, uint8_t hashlength) {
	uint64_t n = 210;
	uint64_t k = 9;

	state->h[0] = blake2b_IV[0] ^ (0x01010000 | hashlength);
	for(int i = 1; i <=5; i++)
		state->h[i] = blake2b_IV[i];

	state->h[6] = blake2b_IV[6] ^ *(reinterpret_cast<const uint64_t *>("AION0PoW"));
	state->h[7] = blake2b_IV[7] ^ ((n << 32) | k);

	// zero bytes processed	
	state->t = 0;
}

void ocl_blake2b::blake2b_update(blake2b_state_t* state, const uint8_t* _m, uint32_t mlength, bool last) {
	uint64_t v[16];

	// treat each 128-bit message chunk as 16 8-byte words, indexed by blake2b_sigma
	const uint64_t* m = reinterpret_cast<const uint64_t*>(_m);

	memcpy(v, state->h, 8*sizeof(*v));
	memcpy(v+8, blake2b_IV, 8*sizeof(*v));
	v[12] ^= (state->t += mlength);
	// v[13] is for hi 64 bits of t
	v[14] ^= last ? 0xFFFFFFFFFFFFFFFF : 0;

	for (int round = 0; round < BLAKE2B_ROUNDS; round++) {
		const uint8_t *s = blake2b_SIGMA[round];
		mix(v, &v[4], &v[8], &v[12], m[s[0]], m[s[1]]);
		mix(&v[1], &v[5], &v[9], &v[13], m[s[2]], m[s[3]]);
		mix(&v[2], &v[6], &v[10], &v[14], m[s[4]], m[s[5]]);
		mix(&v[3], &v[7], &v[11], &v[15], m[s[6]], m[s[7]]);

		mix(&v[0], &v[5], &v[10], &v[15], m[s[8]], m[s[9]]);
		mix(&v[1], &v[6], &v[11], &v[12], m[s[10]], m[s[11]]);
		mix(&v[2], &v[7], &v[8], &v[13], m[s[12]], m[s[13]]);
		mix(&v[3], &v[4], &v[9], &v[14], m[s[14]], m[s[15]]);
		// ...
	}

	// mix upper/lower halves of V into persistent state vector
	for(int i = 0; i<8; i++) {
		state->h[i] ^= v[i] ^ v[i+8];
	}
}

void ocl_blake2b::blake2b_last(blake2b_state_t* state, uint8_t* out, uint8_t outlength) {
	memcpy(out, state->h, outlength);
}
