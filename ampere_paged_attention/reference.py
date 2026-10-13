import math
import torch

def _rope(x, cos, sin):
    if cos is None: return x
    d = x.shape[-1]; half = d // 2
    c, s = cos[..., :half], sin[..., :half]
    a, b = x[..., :half], x[..., half:2*half]
    return torch.cat((a*c-b*s, a*s+b*c, x[..., 2*half:]), -1)

def paged_decode_attention_reference(q, k_pages, v_pages, block_table, context_lens,
                                     k_scale=1.0, v_scale=1.0, cos=None, sin=None,
                                     page_size=16):
    # q [B,Hq,D], pages [NP,Hkv,P,D], int8 K/V, table [B,max_pages].
    B,Hq,D = q.shape; Hkv = k_pages.shape[1]; g = Hq // Hkv
    out = torch.empty_like(q, dtype=torch.float32)
    qf = _rope(q.float(), cos, sin)
    ks = torch.as_tensor(k_scale, device=q.device, dtype=torch.float32)
    vs = torch.as_tensor(v_scale, device=q.device, dtype=torch.float32)
    for b in range(B):
      n = int(context_lens[b]);
      for h in range(Hq):
        kh = h // g; qi = qf[b,h]; m = -float('inf'); l = 0.; acc = torch.zeros(D, device=q.device)
        for t in range(n):
          page = int(block_table[b,t//page_size]); off = t % page_size
          k = k_pages[page,kh,off].float() * ks if ks.ndim == 0 else k_pages[page,kh,off].float()*ks[kh]
          v = v_pages[page,kh,off].float() * vs if vs.ndim == 0 else v_pages[page,kh,off].float()*vs[kh]
          z = torch.dot(qi,k) / math.sqrt(D); nm = max(m, float(z));
          a = math.exp(m-nm) if m != -float('inf') else 0.; p = math.exp(float(z)-nm)
          acc = acc*a + v*p; l = l*a + p; m = nm
        out[b,h] = acc/l
    return out.to(q.dtype)
