"""Ozaki rung 2a (cuda, 3080): the wall-clock race. Slices of s=8
bits are exactly representable in TF32's 11 significant bits and the
tensor-core accumulator is full fp32 — with block<=256 along K,
partial sums stay <= 2^24 = exactly representable: TENSOR CORES AS
EXACT INTEGER UNITS. Race: sliced-exact (full + triangular) vs
native fp64 (rationed 1/64 on gaming cards) vs fp32/TF32.
Error scored against a CPU fp64 reference (itself ~1e-16)."""
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
N = 2048
S, BLOCK = 8, 256
g = torch.Generator().manual_seed(1)
A = (torch.randn(N, N, generator=g) * 0.05).float()
B = (torch.randn(N, N, generator=g) * 0.05).float()
ref = (A.double() @ B.double())        # CPU fp64 reference
scale = ref.abs().max().item()

def slices_of(F, s):
    out, R = [], F
    while R.abs().max() > 0:
        Q = torch.round(R * 2.0**s)
        R = R * 2.0**s - Q
        out.append(Q)
    return out

def sliced_matmul(A, B, s=S, block=BLOCK, tri=None):
    C = torch.zeros(N, N, dtype=torch.float64, device=dev)
    for b0 in range(0, N, block):
        Ab = A[:, b0:b0+block].to(dev).double()
        Bb = B[b0:b0+block, :].to(dev).double()
        ea = (Ab.abs().amax(1, keepdim=True) + 1e-300).log2().floor() + 1
        eb = (Bb.abs().amax(0, keepdim=True) + 1e-300).log2().floor() + 1
        Asl = slices_of(Ab * 2.0**-ea, s)
        Bsl = slices_of(Bb * 2.0**-eb, s)
        for i, Ai in enumerate(Asl):
            Af = Ai.float()
            for j, Bj in enumerate(Bsl):
                if tri is not None and i + j >= tri:
                    continue
                p = (Af @ Bj.float()).double()
                C += p * 2.0**(-s*(i+1)-s*(j+1)) * 2.0**ea * 2.0**eb
    return C

def bench(name, fn, n=3):
    fn()                                # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        C = fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    err = (C.double().cpu() - ref).abs().max().item() / scale
    print(f"  {name:28s} {dt*1e3:8.1f} ms   normwise err {err:.3e}",
          flush=True)

Ad, Bd = A.to(dev), B.to(dev)
A64, B64 = Ad.double(), Bd.double()
bench("fp32/TF32 matmul", lambda: Ad @ Bd)
torch.backends.cuda.matmul.allow_tf32 = False
bench("fp32 strict matmul", lambda: Ad @ Bd)
torch.backends.cuda.matmul.allow_tf32 = True
bench("native fp64 matmul", lambda: A64 @ B64)
bench("sliced EXACT (full k^2)", lambda: sliced_matmul(A, B))
bench("sliced triangular i+j<4", lambda: sliced_matmul(A, B, tri=4))
bench("sliced triangular i+j<3", lambda: sliced_matmul(A, B, tri=3))
