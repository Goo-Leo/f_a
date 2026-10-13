import torch
from ampere_paged_attention.reference import paged_decode_attention_reference
def test_shapes():
    B,Hq,Hkv,D,P,N=2,8,2,32,4,29
    q=torch.randn(B,Hq,D); k=torch.randint(-127,128,(8,Hkv,P,D),dtype=torch.int8); v=torch.randint(-127,128,(8,Hkv,P,D),dtype=torch.int8)
    tab=torch.arange(8).view(2,4); lens=torch.tensor([N,17]); y=paged_decode_attention_reference(q,k,v,tab,lens,.01,.02)
    assert y.shape==q.shape and torch.isfinite(y).all()
if __name__=='__main__': test_shapes(); print('reference OK')
