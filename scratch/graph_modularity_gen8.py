"""Graph-modularity read: gen-8 five-grammar crystal vs single-grammar 19M.

Pre-reg: RESULTS.md 2026-07-26 "PRE-REG: graph-modularity on gen-8".
NEW instrument (07-17 script lost, k unrecorded) — internally paired
only; never compare these numbers to the 07-17 entry.

Construction (fixed in the pre-reg): per-layer kNN graph, k=10,
cosine similarity over FFN neurons (feature = concat of gate-row and
up-row), Newman greedy modularity Q + average clustering, mean over
layers. CPU, minutes.
"""
import networkx as nx
import torch
import torch.nn.functional as F
from networkx.algorithms.community import greedy_modularity_communities, modularity

K = 10


def load(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return sd


def layer_graph(feat: torch.Tensor) -> nx.Graph:
    x = F.normalize(feat, dim=1)
    sim = x @ x.T
    sim.fill_diagonal_(-2.0)
    idx = sim.topk(K, dim=1).indices
    g = nx.Graph()
    g.add_nodes_from(range(feat.shape[0]))
    for i in range(feat.shape[0]):
        for j in idx[i].tolist():
            g.add_edge(i, j)
    return g


def read(path: str) -> tuple[float, float]:
    sd = load(path)
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    qs, cs = [], []
    for li in layers:
        feat = torch.cat(
            [sd[f"blocks.{li}.gate.weight"], sd[f"blocks.{li}.up.weight"]], dim=1
        ).float()
        g = layer_graph(feat)
        comms = greedy_modularity_communities(g)
        qs.append(modularity(g, comms))
        cs.append(nx.average_clustering(g))
        print(f"  layer {li}: Q={qs[-1]:.4f} clustering={cs[-1]:.4f} "
              f"n_comms={len(comms)}", flush=True)
    return sum(qs) / len(qs), sum(cs) / len(cs)


if __name__ == "__main__":
    arms = {
        "A_gen8_five_grammar": "checkpoints/mathnative_19m_gen8.pt",
        "B_single_grammar_19m": "checkpoints/mathnative_19m.pt",
    }
    out = {}
    for name, path in arms.items():
        print(f"== {name}: {path}")
        out[name] = read(path)
    for name, (q, c) in out.items():
        print(f"{name}: mean Q={q:.4f} mean clustering={c:.4f}")
    dq = out["A_gen8_five_grammar"][0] - out["B_single_grammar_19m"][0]
    print(f"delta Q (A-B) = {dq:+.4f}  (pre-reg: modules iff > +0.05)")
