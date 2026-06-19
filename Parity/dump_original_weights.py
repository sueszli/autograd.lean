# /// script
# requires-python = "==3.14.*"
# ///
import json
import math
import random
from operator import mul
from pathlib import Path

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4

DATA_IN = Path(__file__).resolve().parent.parent / "data" / "input.txt"
DATA_OUT = Path(__file__).resolve().parent.parent / "data" / "parity.json"


#
# kernels
#


# Plain accumulator. CPython 3.12+ builtin sum() uses Neumaier compensation
# for float-only inputs; original.py routes adds through Value autograd, which
# never hits that path. Divisions are written `x * (1/y)` for the same reason
# (mirrors Value.__truediv__'s `self * other**-1`).
def fsum(it):
    s = 0.0
    for v in it:
        s = s + v
    return s


def linear(x, w):
    return [fsum(map(mul, wo, x)) for wo in w]


def softmax(v):
    m = max(v)
    e = [math.exp(z - m) for z in v]
    inv_s = 1.0 / fsum(e)
    return [z * inv_s for z in e]


def rmsnorm(x):
    inv_d = 1.0 / len(x)
    scale = (fsum(z * z for z in x) * inv_d + 1e-5) ** -0.5
    return [z * scale for z in x]


#
# model
#


def gpt(token_id, pos_id, keys, values):
    head_dim = N_EMBED // N_HEAD
    inv_scale = 1.0 / head_dim**0.5
    x = [a + b for a, b in zip(sd["wte"][token_id], sd["wpe"][pos_id])]
    x = rmsnorm(x)
    for li in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, sd[f"layer{li}.attn_wq"])
        k = linear(x, sd[f"layer{li}.attn_wk"])
        v = linear(x, sd[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * head_dim
            qh = q[hs : hs + head_dim]
            kh = [ki[hs : hs + head_dim] for ki in keys[li]]
            vh = [vi[hs : hs + head_dim] for vi in values[li]]
            scores = [fsum(map(mul, qh, kh[t])) * inv_scale for t in range(len(kh))]
            aw = softmax(scores)
            x_attn.extend(fsum(aw[t] * vh[t][j] for t in range(len(vh))) for j in range(head_dim))
        x = linear(x_attn, sd[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, sd[f"layer{li}.mlp_fc1"])
        x = [v if v > 0.0 else 0.0 for v in x]
        x = linear(x, sd[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]
    return linear(x, sd["lm_head"])


#
# init + one forward + dump
#


docs = [d for d in DATA_IN.read_text().splitlines() if d]
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

matrix = lambda nout, nin, std=0.08: [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]
sd = {
    "wte": matrix(vocab_size, N_EMBED),
    "wpe": matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": matrix(vocab_size, N_EMBED),
    **{f"layer{i}.attn_wq": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

doc = docs[0]
tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
n = min(BLOCK_SIZE, len(tokens) - 1)

keys = [[] for _ in range(N_LAYER)]
values = [[] for _ in range(N_LAYER)]
logits_per_pos = []
losses = []
for pos_id in range(n):
    logits = gpt(tokens[pos_id], pos_id, keys, values)
    logits_per_pos.append(logits)
    losses.append(-math.log(softmax(logits)[tokens[pos_id + 1]]))
loss = fsum(losses) * (1.0 / n)

DATA_OUT.write_text(
    json.dumps(
        {
            "vocab_size": vocab_size,
            "bos": BOS,
            "tokens": tokens,
            "n": n,
            "logits": logits_per_pos,
            "loss": loss,
            "weights": sd,
        }
    )
)
print(f"wrote {DATA_OUT} doc={doc!r} n={n} loss={loss:.6f}")
