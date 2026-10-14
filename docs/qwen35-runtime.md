# Qwen3.5 stateful runtime support and decode optimization

Adds recurrent/conv-state feedback, inferred graph specifications and quantized boundaries for Qwen3.5 text models, including split-transformer and fullgraph contexts. No model binaries or QAIRT SDK are included in this change.

Greedy decoding without grammar or penalties uses encoded-logit argmax; the uint16 ARM path uses NEON and preserves the first index on ties. Conv feedback copies valid tails directly. Independent KV work is submitted before recurrent feedback; recurrent pairs can be divided among existing workers. Pending work is drained on exceptional exits.

Optional controls used on the QCS8550 benchmark:

```sh
export GENIEX_ALLOW_QUANT_BOUNDARY=1
export GENIEX_QNN_LOG_WARN=1
export GENIEX_DECODE_WORKERS=3
export GENIEX_DECODE_CPUMASK=58 # hexadecimal CPU mask: CPUs 3,4,6
export GENIEX_DECODE_POLL=1
export GENIEX_CLOCK_KEEPER_THREADS=0
export GENIEX_RECURRENT_WORKERS=3
export GENIEX_EXACT_STATE_ZERO=1
export GENIEX_DIRECT_QUANT_MASK=1
export GENIEX_AR1_PREFILL_TAIL=1
export GENIEX_DEFER_DECODE_RECURRENT_PREFILL=1
export GENIEX_DEFER_PREFILL_RECURRENT=1 # optional deferred AR128 mirrors during decode
```

Pin the calling main thread to CPU4 (`taskset 10` on Android), and explicitly use GenieX temperature=-1 for greedy decoding. CPU masks are specific to the tested board; they are not portable defaults. Deferred feedback still updates the active graph state before its next execution; only inactive graph mirrors are postponed.

For direct-stock contexts whose configuration explicitly declares concat RoPE, use `GENIEX_CONCAT_OFFICIAL_QUANT=1` to match the tested official quantization formula. It must not be applied as a universal numerical policy. Model positional-encoding configuration must match its export.

`GENIEX_AR1_PREFILL_TAIL` handles one remaining prompt token through AR1 after wider prefill, matching the tested official path. Full-logits/PPL hooks retain fixed-width prefill behavior. Exact-zero initialization avoids floating-point truncation of the quantized recurrent-state zero code.

## Validation on 2026-10-14

Same direct-stock V73/VTCM8 context, QAIRT 2.51.40.260930, identical random token IDs (Android rand, seed43), state reset, no warmup, 2048 input and actual512 output, three runs:

| Runtime | Decode tok/s | Input tokens / TTFT seconds |
|---|---:|---:|
| Official Genie 1.21.0, same CPU affinity | 37.40 | 1766 |
| GenieX, deferred mirrors | 37.57 | 1742 |

All512 output token IDs match the official reference. Additional3072/512 and129/64 tests match token IDs, including ordinary feedback. Standalone checks covered NEON ties/tails/extremes/random vocabulary and thread-pool barriers/errors/shutdown. This is runtime equivalence on those inputs, not a new PPL or QA accuracy claim. Virtual battery reporting was corrected for the benchmark; real thermal protections were retained.

Model context SHA256: `eb9a45ecc0572200bf54f4ed5b09c6147e520e769d9e176d545f5446b667725b`.

Build normally with the repository submodules. `GENIEX_PREBUILT_PROC` is an optional CMake path for reusing a matching unchanged text processor shared library; VLM reuse was not validated. Leave it unset for a normal source build.
