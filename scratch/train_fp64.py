"""fp64 end-to-end birth (the rounding-loss-veil A/B, banked
2026-07-17): all weights/activations/optimizer double precision on
CPU. One variable vs seedvar-1 (fp32, same seed/diet). If the gate
moves >=3, matmul/update rounding at fp32 costs capability at birth
— the veil is real. If flat, fp32 birth arithmetic is above the
noise floor and precision stays an ONLINE-only knob."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
torch.set_default_dtype(torch.float64)
# force CPU: MPS has no fp64
torch.backends.mps.is_available = lambda: False
import train_mathnative as T
T.main(gen4=True, epochs=3, out="checkpoints/fp64_birth.pt")
