/**
 * trn_resonance.c — Pure C TRN resonance layer implementation.
 *
 * Exactly replicates Python TemporalResonanceLayer.step_single() arithmetic:
 *
 *   proj(x)   -> [A_raw | Om_raw | Ph_raw | Ga_raw]   (single Linear, bias=True)
 *   A         = softplus(A_raw).clamp(max=amplitude_max)
 *   omega     = sigmoid(Om_raw) * PI + omega_base[k]
 *   phi       = tanh(Ph_raw) * PI
 *   alpha     = sigmoid(Ga_raw)
 *
 *   pos       = log1p(position)   if phase_mode==0 (log)
 *               position          if phase_mode==1 (linear)
 *   angle     = omega * pos + phi
 *
 *   one_m_a   = 1 - alpha
 *   v_r       = one_m_a * A * cos(angle)
 *   v_i       = one_m_a * A * sin(angle)
 *
 *   r_real    = alpha * r_real + v_r         (state update, fp32)
 *   r_imag    = alpha * r_imag + v_i
 *
 *   if state_norm:
 *     scale   = max(|r_real|, |r_imag|).clamp(min=1.0)   (per-channel)
 *     r_real /= scale
 *     r_imag /= scale
 *
 *   rho       = r_real * cos(angle) + r_imag * sin(angle)   (demodulate)
 *   out       = res_scale * (w_res_weight @ rho)             (output projection)
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o trn_resonance.so trn_resonance.c -lm
 */

#include "trn_resonance.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- constants ---- */

#define TRN_MAGIC   0x54524E31u
#define TRN_VERSION 1u

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- internal structure ---- */

struct TRNResonanceLayer {
    int   d_model;
    int   K;
    int   phase_mode;    /* 0 = log, 1 = linear */
    int   state_norm;    /* 0 = off, 1 = on */
    float amplitude_max;
    float res_scale;

    /* proj: [4K, d_model] weight + [4K] bias */
    float * proj_weight;   /* [4K * d_model] */
    float * proj_bias;     /* [4K] */

    /* learnable base frequencies: [K] */
    float * omega_base;

    /* W_res: [d_model, K] */
    float * w_res_weight;  /* [d_model * K] */
};

/* ---- scalar activation functions ---- */

/* softplus(x) = log(1 + exp(x)), numerically safe for large x */
static inline float trn_softplus(float x)
{
    if (x > 20.0f) return x;           /* log1p(exp(x)) ≈ x for x > 20 */
    return log1pf(expf(x));
}

/* sigmoid(x) = 1 / (1 + exp(-x)) */
static inline float trn_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/* ---- matrix-vector multiply (no BLAS) ---- */

/*
 * sgemv: y = W * x + b
 *   W shape: [out_dim, in_dim]  (row-major)
 *   x shape: [in_dim]
 *   b shape: [out_dim]  (may be NULL to skip bias)
 *   y shape: [out_dim]
 */
static void sgemv(
    const float * W,
    const float * x,
    const float * b,
    float       * y,
    int           out_dim,
    int           in_dim
) {
    for (int o = 0; o < out_dim; ++o) {
        float acc = (b != NULL) ? b[o] : 0.0f;
        const float * row = W + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
}

/* ---- file loading ---- */

/* Read all bytes from path into a malloc'd buffer; *out_size is set. */
static uint8_t * read_file_bytes(const char * path, size_t * out_size)
{
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    rewind(f);

    uint8_t * buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }

    if ((size_t)sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

/* ---- public API ---- */

TRNResonanceLayer * trn_resonance_load(const char * path)
{
    size_t   file_size = 0;
    uint8_t * raw      = read_file_bytes(path, &file_size);
    if (!raw) {
        fprintf(stderr, "trn_resonance_load: cannot read '%s'\n", path);
        return NULL;
    }

    /* --- parse header (32 bytes = 8 x uint32/float) --- */
    if (file_size < 32) {
        fprintf(stderr, "trn_resonance_load: file too small (%zu bytes)\n", file_size);
        free(raw);
        return NULL;
    }

    uint32_t hdr[8];
    memcpy(hdr, raw, 32);

    uint32_t magic   = hdr[0];
    uint32_t version = hdr[1];
    int32_t  d_model_i, K_i, phase_mode_i, state_norm_i;
    memcpy(&d_model_i,    &hdr[2], 4);
    memcpy(&K_i,          &hdr[3], 4);
    memcpy(&phase_mode_i, &hdr[4], 4);
    memcpy(&state_norm_i, &hdr[5], 4);
    float amplitude_max, res_scale;
    memcpy(&amplitude_max, &hdr[6], 4);
    memcpy(&res_scale,     &hdr[7], 4);

    if (magic != TRN_MAGIC) {
        fprintf(stderr, "trn_resonance_load: bad magic 0x%08X (expected 0x%08X)\n",
                magic, TRN_MAGIC);
        free(raw);
        return NULL;
    }
    if (version != TRN_VERSION) {
        fprintf(stderr, "trn_resonance_load: unsupported version %u\n", version);
        free(raw);
        return NULL;
    }

    int d_model    = (int)d_model_i;
    int K          = (int)K_i;
    int phase_mode = (int)phase_mode_i;
    int state_norm = (int)state_norm_i;

    /* --- verify file size --- */
    size_t n_proj_w  = (size_t)(4 * K) * d_model;
    size_t n_proj_b  = (size_t)(4 * K);
    size_t n_omega   = (size_t)K;
    size_t n_wres    = (size_t)d_model * K;
    size_t n_total   = 32 + (n_proj_w + n_proj_b + n_omega + n_wres) * sizeof(float);

    if (file_size != n_total) {
        fprintf(stderr,
                "trn_resonance_load: size mismatch — got %zu, expected %zu "
                "(d_model=%d K=%d)\n",
                file_size, n_total, d_model, K);
        free(raw);
        return NULL;
    }

    /* --- allocate struct and copy weights --- */
    TRNResonanceLayer * layer = (TRNResonanceLayer *)calloc(1, sizeof(TRNResonanceLayer));
    if (!layer) { free(raw); return NULL; }

    layer->d_model       = d_model;
    layer->K             = K;
    layer->phase_mode    = phase_mode;
    layer->state_norm    = state_norm;
    layer->amplitude_max = amplitude_max;
    layer->res_scale     = res_scale;

    layer->proj_weight  = (float *)malloc(n_proj_w * sizeof(float));
    layer->proj_bias    = (float *)malloc(n_proj_b * sizeof(float));
    layer->omega_base   = (float *)malloc(n_omega  * sizeof(float));
    layer->w_res_weight = (float *)malloc(n_wres   * sizeof(float));

    if (!layer->proj_weight || !layer->proj_bias ||
        !layer->omega_base  || !layer->w_res_weight) {
        trn_resonance_free(layer);
        free(raw);
        return NULL;
    }

    const float * data = (const float *)(raw + 32);
    memcpy(layer->proj_weight,  data,                                    n_proj_w * sizeof(float));
    memcpy(layer->proj_bias,    data + n_proj_w,                         n_proj_b * sizeof(float));
    memcpy(layer->omega_base,   data + n_proj_w + n_proj_b,              n_omega  * sizeof(float));
    memcpy(layer->w_res_weight, data + n_proj_w + n_proj_b + n_omega,    n_wres   * sizeof(float));

    free(raw);
    return layer;
}

void trn_resonance_free(TRNResonanceLayer * layer)
{
    if (!layer) return;
    free(layer->proj_weight);
    free(layer->proj_bias);
    free(layer->omega_base);
    free(layer->w_res_weight);
    free(layer);
}

float * trn_resonance_state_alloc(const TRNResonanceLayer * layer, int batch_size)
{
    if (!layer || batch_size <= 0) return NULL;
    /* r_real and r_imag interleaved: [batch, K, 2] */
    size_t n = (size_t)batch_size * layer->K * 2;
    float * state = (float *)calloc(n, sizeof(float));
    return state;
}

void trn_resonance_state_free(float * state)
{
    free(state);
}

int trn_resonance_step(
    const TRNResonanceLayer * layer,
    const float             * x,
    float                   * r_real,
    float                   * r_imag,
    int                       position,
    float                   * out,
    int                       batch_size
) {
    if (!layer || !x || !r_real || !r_imag || !out || batch_size <= 0)
        return -1;

    const int d = layer->d_model;
    const int K = layer->K;

    /* Temporary buffer for projection output: [4K] per sample */
    float * proj_out = (float *)malloc((size_t)(4 * K) * sizeof(float));
    /* Temporary buffer for rho: [K] per sample */
    float * rho_buf  = (float *)malloc((size_t)K * sizeof(float));

    if (!proj_out || !rho_buf) {
        free(proj_out);
        free(rho_buf);
        return -2;
    }

    /* position encoding */
    float pos_enc;
    if (layer->phase_mode == 0) {          /* log mode: log1p(position) */
        pos_enc = log1pf((float)position);
    } else {                                /* linear mode: position */
        pos_enc = (float)position;
    }

    for (int b = 0; b < batch_size; ++b) {
        const float * x_b     = x      + (size_t)b * d;
        float       * r_real_b = r_real + (size_t)b * K;
        float       * r_imag_b = r_imag + (size_t)b * K;
        float       * out_b    = out    + (size_t)b * d;

        /* --- projection: proj_out = proj_weight @ x_b + proj_bias --- */
        sgemv(layer->proj_weight, x_b, layer->proj_bias, proj_out, 4 * K, d);

        /*
         * proj_out layout:
         *   [0  .. K-1]   A_raw
         *   [K  .. 2K-1]  Om_raw
         *   [2K .. 3K-1]  Ph_raw
         *   [3K .. 4K-1]  Ga_raw
         */
        const float * A_raw  = proj_out;
        const float * Om_raw = proj_out + K;
        const float * Ph_raw = proj_out + 2 * K;
        const float * Ga_raw = proj_out + 3 * K;

        /* --- per-oscillator computation --- */
        for (int k = 0; k < K; ++k) {
            /* activations — match Python exactly */
            float A     = trn_softplus(A_raw[k]);
            if (A > layer->amplitude_max) A = layer->amplitude_max;

            float omega = trn_sigmoid(Om_raw[k]) * (float)M_PI + layer->omega_base[k];
            float phi   = tanhf(Ph_raw[k]) * (float)M_PI;
            float alpha = trn_sigmoid(Ga_raw[k]);

            /* angle = omega * pos_enc + phi */
            float angle   = omega * pos_enc + phi;
            float cos_ang = cosf(angle);
            float sin_ang = sinf(angle);

            float one_m_a = 1.0f - alpha;
            float v_r     = one_m_a * A * cos_ang;
            float v_i     = one_m_a * A * sin_ang;

            /* state update (fp32) */
            float new_r_real = alpha * r_real_b[k] + v_r;
            float new_r_imag = alpha * r_imag_b[k] + v_i;

            /* state normalisation (per-channel max-abs) */
            if (layer->state_norm) {
                float abs_r = fabsf(new_r_real);
                float abs_i = fabsf(new_r_imag);
                float scale = (abs_r > abs_i) ? abs_r : abs_i;
                if (scale < 1.0f) scale = 1.0f;
                new_r_real /= scale;
                new_r_imag /= scale;
            }

            r_real_b[k] = new_r_real;
            r_imag_b[k] = new_r_imag;

            /* demodulate */
            rho_buf[k] = new_r_real * cos_ang + new_r_imag * sin_ang;
        }

        /* --- output projection: out_b = res_scale * (w_res_weight @ rho_buf) --- */
        /* w_res_weight shape: [d_model, K]  ->  sgemv with NULL bias */
        sgemv(layer->w_res_weight, rho_buf, NULL, out_b, d, K);
        const float rs = layer->res_scale;
        for (int i = 0; i < d; ++i) {
            out_b[i] *= rs;
        }
    }

    free(proj_out);
    free(rho_buf);
    return 0;
}

/* ---- metadata accessors ---- */

int   trn_resonance_d_model(const TRNResonanceLayer * l)     { return l ? l->d_model       : -1; }
int   trn_resonance_K(const TRNResonanceLayer * l)           { return l ? l->K              : -1; }
int   trn_resonance_phase_mode(const TRNResonanceLayer * l)  { return l ? l->phase_mode     : -1; }
int   trn_resonance_state_norm(const TRNResonanceLayer * l)  { return l ? l->state_norm     :  0; }
float trn_resonance_amplitude_max(const TRNResonanceLayer * l){ return l ? l->amplitude_max : 0.0f; }
float trn_resonance_res_scale(const TRNResonanceLayer * l)   { return l ? l->res_scale     : 0.0f; }
