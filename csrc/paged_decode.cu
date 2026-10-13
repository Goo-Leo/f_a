#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__global__ void decode_kernel(const scalar_t* q,const int8_t* K,const int8_t* V,float* O,
 const int32_t* table,const int32_t* lens,const float* ks,const float* vs,const float* rc,const float* rs,
 int B,int Hq,int Hkv,int P,int D,int maxp,float invsqrt) {
 int h=blockIdx.x, b=blockIdx.y, tid=threadIdx.x, g=Hq/Hkv, kh=h/g, n=lens[b];
 if (h>=Hq||b>=B) return;
 for(int d=tid;d<D;d+=blockDim.x) O[(b*Hq+h)*D+d]=0.f;
 __shared__ float sm[32];
 float m=-CUDART_INF_F, l=0.f;
 for(int t=0;t<n;t++) {
   int page=table[b*maxp+t/P], off=t%P, base=((page*Hkv+kh)*P+off)*D;
   float dot=0.f;
   for(int d=tid;d<D;d+=blockDim.x) { float qv=float(q[(b*Hq+h)*D+d]); float kv=float(K[base+d])*ks[kh]; if(rc){int j=d%(D/2); float c=rc[j],s=rs[j]; if(d<D/2){float q2=float(q[(b*Hq+h)*D+d+D/2]);float k2=float(K[base+d+D/2])*ks[kh];qv=qv*c-q2*s;kv=kv*c-k2*s;} else {float q2=float(q[(b*Hq+h)*D+d-D/2]);float k2=float(K[base+d-D/2])*ks[kh];qv=qv*c+q2*s;kv=kv*c+k2*s;}} dot += qv*kv; }
   sm[tid]=dot; __syncthreads();
   for(int w=16;w;w>>=1) sm[tid]+=__shfl_down_sync(0xffffffff,sm[tid],w);
   if(tid==0) sm[0]*=invsqrt;
   __syncthreads(); float z=sm[0], nm=fmaxf(m,z), a=(m==-CUDART_INF_F?0.f:expf(m-nm)), p=expf(z-nm);
   for(int d=tid;d<D;d+=blockDim.x) { float old=O[(b*Hq+h)*D+d]; O[(b*Hq+h)*D+d]=old*a+float(V[base+d])*vs[kh]*p; }
   l=l*a+p; m=nm; __syncthreads();
 }
 for(int d=tid;d<D;d+=blockDim.x) O[(b*Hq+h)*D+d]/=l;
}

torch::Tensor forward(torch::Tensor q,torch::Tensor K,torch::Tensor V,torch::Tensor table,
 torch::Tensor lens,torch::Tensor ks,torch::Tensor vs, c10::optional<torch::Tensor> cos,
 c10::optional<torch::Tensor> sin,int64_t P) {
 TORCH_CHECK(q.is_cuda()&&K.scalar_type()==torch::kChar&&V.scalar_type()==torch::kChar,"invalid tensors");
 auto o=torch::empty_like(q, q.options().dtype(torch::kFloat)); int B=q.size(0),H=q.size(1),D=q.size(2),HK=K.size(1);
 auto k=K.contiguous(),v=V.contiguous(); auto s=ks.numel()==1?ks.expand({HK}):ks; auto vs2=vs.numel()==1?vs.expand({HK}):vs; const float* rc=cos?cos->data_ptr<float>():nullptr; const float* rs=sin?sin->data_ptr<float>():nullptr;
 AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(),"paged_decode",([&]{decode_kernel<scalar_t><<<dim3(H,B),32>>>(q.data_ptr<scalar_t>(),k.data_ptr<int8_t>(),v.data_ptr<int8_t>(),o.data_ptr<float>(),table.data_ptr<int32_t>(),lens.data_ptr<int32_t>(),s.data_ptr<float>(),vs2.data_ptr<float>(),rc,rs,B,H,HK,P,D,table.size(1),rsqrtf(float(D)));}));
 return o.to(q.scalar_type());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
