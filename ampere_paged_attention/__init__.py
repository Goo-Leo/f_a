from .reference import paged_decode_attention_reference
try:
    from . import _C
except ImportError:
    _C = None

def paged_decode_attention(q, k_pages, v_pages, block_table, context_lens, k_scale=1.0,
                           v_scale=1.0, cos=None, sin=None, page_size=16):
    """Fused CUDA path; falls back to the numerically clear reference implementation."""
    if _C is None or not q.is_cuda:
        return paged_decode_attention_reference(q, k_pages, v_pages, block_table,
                                                context_lens, k_scale, v_scale, cos, sin, page_size)
    return _C.forward(q, k_pages, v_pages, block_table, context_lens, k_scale, v_scale,
                      cos, sin, page_size)
