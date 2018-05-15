
namespace ocl_blake2b {

typedef struct blake2b_state {
	uint64_t h[8];
	uint64_t t;  // not 128 bits
} blake2b_state_t;

void blake2b_init(blake2b_state_t* state, uint8_t hashlength);
void blake2b_update(blake2b_state_t* state, const uint8_t* _m, uint32_t mlength, bool last);
void blake2b_last(blake2b_state_t* state, uint8_t* out, uint8_t outlength);

}  // ocl_blake2b
