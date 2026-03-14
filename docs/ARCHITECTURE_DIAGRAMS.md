# Architecture Diagrams

## 1. Dual Memory Architecture

```
                         Input Tokens
                              |
                              v
                    +-------------------+
                    |    Embedding      |
                    +-------------------+
                              |
                              v
              +-------------------------------+
              |        Hybrid Block N         |
              |                               |
              |  +----------+  +----------+   |
              |  | KV Window |  | TRN State|   |
              |  | (recent)  |  | (distant)|   |
              |  +----------+  +----------+   |
              |       |              |         |
              |       v              v         |
              |  +--------------------------+  |
              |  |     Learned Mixer Gate   |  |
              |  |  g = sigmoid(W_g * x)   |  |
              |  |  out = g*attn + (1-g)*trn|  |
              |  +--------------------------+  |
              |              |                 |
              +-------------------------------+
                              |
                              v
                    +-------------------+
                    |   RMSNorm + LM    |
                    +-------------------+
                              |
                              v
                        Next Token


    Memory Comparison:

    KV Window (short-term):       TRN State (long-term):
    +---+---+---+---+---+        +-------+-------+
    | K | K | K | K | K |        | r_real| r_imag|
    +---+---+---+---+---+        +-------+-------+
    | V | V | V | V | V |        | (1,K) | (1,K) |
    +---+---+---+---+---+        +-------+-------+
    Grows with T: O(n)           Fixed: O(1)
    n_layers * 2 * H * T * d_h   n_layers * K * 2 * 4
```

## 2. KV Eviction to TRN Compression

```
    Time --->

    Token stream:   t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 ...
                    |---|---|---|---|---|---|---|---|---|---|

    KV Window (size W=4):
    +----+----+----+----+
    | t7 | t8 | t9 | t10|  <-- Active KV cache (recent W tokens)
    +----+----+----+----+
         ^
         |  Eviction
         |
    +----+----+----+
    | t4 | t5 | t6 |  <-- Evicted tokens (would be discarded)
    +----+----+----+
         |
         v
    TRN State Update:
    +------------------------------------------+
    | r_t = alpha * r_{t-1} + v_t              |
    |                                          |
    | v_t = (1-alpha) * A * exp(j*(w*t + phi)) |
    |                                          |
    | State: r_real(1,K) + r_imag(1,K)         |
    | Size:  K * 2 * 4 bytes = CONSTANT        |
    +------------------------------------------+
         |
         v
    TRN State (compressed history):
    +-------+-------+
    | r_real| r_imag|   Contains information from t1..t6
    +-------+-------+   (lossy compression, not exact recall)
    | 16 KB | fixed |
    +-------+-------+


    Without TRN:  t1..t6 discarded entirely
    With TRN:     t1..t6 compressed into constant-size state
    Tradeoff:     No exact recall, but statistical patterns preserved
```

## 3. Multi-Agent Scaling: O(1) vs O(n)

```
    KV Cache: O(n) per agent            TRN State: O(1) per agent
    ========================            =========================

    Agent 1: [KV: 31.25 MB]            Agent 1: [State: 16 KB]
    Agent 2: [KV: 31.25 MB]            Agent 2: [State: 16 KB]
    Agent 3: [KV: 31.25 MB]            Agent 3: [State: 16 KB]
       ...                                ...
    Agent N: [KV: 31.25 MB]            Agent N: [State: 16 KB]
    ________________________            _________________________
    Total:   N * 31.25 MB              Total:   N * 16 KB


    Example: trn_100m, T=1000, 8 heads, head_dim=64, fp32 K+V

    N agents  |  KV Total    |  TRN Total  |  Ratio
    ----------|--------------|-------------|--------
    10        |  312 MB      |  0.16 MB    |  2,000x
    100       |  3,125 MB    |  1.56 MB    |  2,000x
    1,000     |  31,250 MB   |  15.6 MB    |  2,000x
    10,000    |  312,500 MB  |  156 MB     |  2,000x


    GPU Capacity (A100 80GB):

    KV Cache:
    +========================================+
    |  80 GB / 31.25 MB = 2,560 agents max   |
    +========================================+

    TRN State:
    +========================================+
    |  80 GB / 16 KB = 5,242,880 agents max  |
    +========================================+

    Note: These are state-only numbers. Model weights (~150 MB for
    trn_100m) are shared across all agents and not counted here.
    Actual agent capacity is lower due to model weight memory.
```

## 4. TRN Resonance Layer (Single Step)

```
    Input: x_t (B, d_model)
                |
                v
    +-------------------------+
    | OscillatorProjection    |
    | x -> A, omega, phi, alpha|
    |                         |
    | A     = softplus(W_a*x) |  amplitude (clamped <= 3.0)
    | omega = W_w * x         |  frequency
    | phi   = W_p * x         |  phase
    | alpha = sigmoid(W_g*x)  |  decay gate (init ~0.7)
    +-------------------------+
                |
                v
    +-------------------------+
    | Complex Recurrence      |
    |                         |
    | theta = omega * log(t+1)|  log-phase encoding
    | v_real = (1-a)*A*cos(theta+phi)
    | v_imag = (1-a)*A*sin(theta+phi)
    |                         |
    | r_real = a*r_real + v_real  <-- state update
    | r_imag = a*r_imag + v_imag  <-- O(1) per step
    +-------------------------+
                |
                v
    +-------------------------+
    | Output Projection       |
    |                         |
    | y = W_out * r_real      |  (B, d_model)
    | y = y * res_scale       |  (init 0.05)
    +-------------------------+
                |
                v
    Output: y_t (B, d_model)

    State: r_real(B,K) + r_imag(B,K) = 2*K*4 bytes per layer
    No dependence on sequence length T.
```
