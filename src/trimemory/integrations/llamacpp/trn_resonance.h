/**
 * trn_resonance.h — Pure C TRN resonance layer (llama.cpp integration style)
 *
 * Binary weight format (little-endian float32):
 *   Header (32 bytes):
 *     [0]  magic:         0x54524E31 (uint32)
 *     [1]  version:       1          (uint32)
 *     [2]  d_model:       int32
 *     [3]  K:             int32
 *     [4]  phase_mode:    int32  (0=log, 1=linear)
 *     [5]  state_norm:    int32  (0=off, 1=on)
 *     [6]  amplitude_max: float32
 *     [7]  res_scale:     float32
 *   Data (all float32, row-major):
 *     proj_weight[4K * d_model]
 *     proj_bias  [4K]
 *     omega_base [K]
 *     w_res_weight[d_model * K]
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o trn_resonance.so trn_resonance.c -lm
 *
 * No external dependencies (stdlib.h, math.h, string.h, stdint.h only).
 */

#ifndef TRN_RESONANCE_H
#define TRN_RESONANCE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- opaque handle ---- */
typedef struct TRNResonanceLayer TRNResonanceLayer;

/* ---- lifecycle ---- */

/**
 * Load a TRN resonance layer from a binary weight file.
 * Returns NULL on failure.
 */
TRNResonanceLayer * trn_resonance_load(const char * path);

/**
 * Free all resources associated with the layer.
 */
void trn_resonance_free(TRNResonanceLayer * layer);

/* ---- state ---- */

/**
 * Allocate zero-initialised resonance state for batch_size sequences.
 * Returns NULL on allocation failure.
 * Caller owns the returned pointer; free with trn_resonance_state_free().
 */
float * trn_resonance_state_alloc(const TRNResonanceLayer * layer,
                                   int batch_size);

/**
 * Free state memory returned by trn_resonance_state_alloc().
 */
void trn_resonance_state_free(float * state);

/* ---- forward (single step) ---- */

/**
 * Execute one autoregressive step.
 *
 * Parameters:
 *   layer    - loaded layer handle
 *   x        - input  [batch_size * d_model] row-major float32
 *   r_real   - in/out [batch_size * K]       float32 (real part of state)
 *   r_imag   - in/out [batch_size * K]       float32 (imag part of state)
 *   position - absolute token index (0-based)
 *   out      - output [batch_size * d_model] float32
 *   batch_size - number of sequences in this call
 *
 * Returns 0 on success, non-zero on error.
 */
int trn_resonance_step(
    const TRNResonanceLayer * layer,
    const float             * x,
    float                   * r_real,
    float                   * r_imag,
    int                       position,
    float                   * out,
    int                       batch_size
);

/* ---- metadata accessors ---- */

int   trn_resonance_d_model(const TRNResonanceLayer * layer);
int   trn_resonance_K(const TRNResonanceLayer * layer);
int   trn_resonance_phase_mode(const TRNResonanceLayer * layer); /* 0=log, 1=linear */
int   trn_resonance_state_norm(const TRNResonanceLayer * layer); /* 0=off, 1=on */
float trn_resonance_amplitude_max(const TRNResonanceLayer * layer);
float trn_resonance_res_scale(const TRNResonanceLayer * layer);

#ifdef __cplusplus
}
#endif

#endif /* TRN_RESONANCE_H */
