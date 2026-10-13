# Ampere GQA paged KV decode attention

This directory contains a small, reproducible implementation of decode attention for
NVIDIA Ampere (SM80, including RTX 3090).  The public API is
`paged_decode_attention(q, k_pages, v_pages, block_table, context_lens, ...)`.
KV pages are int8 and are dequantized inside the CUDA attention loop; RoPE is applied
to the query and keys and the softmax is accumulated online.  A PyTorch reference is
included for correctness checks on machines without CUDA.

The CUDA path is intentionally single-pass over each page.  It supports GQA by
mapping query head `hq` to KV head `hq // (Hq/Hkv)`, and uses one CTA per `(batch,
query-head)` for short decode queries.  Build on a CUDA/PyTorch environment with:

```bash
python -m pip install -e .
python tests/test_reference.py
python tests/test_cuda.py       # on an SM80 CUDA host
```

The kernel expects contiguous tensors and per-KV-head (or scalar) scales.  It is a
benchmark kernel, not a replacement for a production library's autotuned variants.
