# /// script
# requires-python = "==3.14.*"
# ///
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_IN = ROOT / "data" / "input.txt"
DATA_OUT = ROOT / "data" / "parity.json"

N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4

random.seed(42)

docs = [d for d in DATA_IN.read_text().splitlines() if d]
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1


def matrix(nout, nin, std=0.08):
    return [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]


state_dict = {
    "wte": matrix(vocab_size, N_EMBED),
    "wpe": matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": matrix(vocab_size, N_EMBED),
}
for i in range(N_LAYER):
    state_dict[f"layer{i}.attn_wq"] = matrix(N_EMBED, N_EMBED)
    state_dict[f"layer{i}.attn_wk"] = matrix(N_EMBED, N_EMBED)
    state_dict[f"layer{i}.attn_wv"] = matrix(N_EMBED, N_EMBED)
    state_dict[f"layer{i}.attn_wo"] = matrix(N_EMBED, N_EMBED)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * N_EMBED, N_EMBED)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(N_EMBED, 4 * N_EMBED)


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    m = max(logits)
    e = [math.exp(z - m) for z in logits]
    s = sum(e)
    return [v / s for v in e]


def rmsnorm(x):
    ms = sum(z * z for z in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [z * scale for z in x]


def gpt(token_id, pos_id, keys, values):
    x = [a + b for a, b in zip(state_dict["wte"][token_id], state_dict["wpe"][pos_id])]
    x = rmsnorm(x)
    for li in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * N_EMBED // N_HEAD
            qh = q[hs : hs + N_EMBED // N_HEAD]
            kh = [ki[hs : hs + N_EMBED // N_HEAD] for ki in keys[li]]
            vh = [vi[hs : hs + N_EMBED // N_HEAD] for vi in values[li]]
            logits = [sum(qh[j] * kh[t][j] for j in range(N_EMBED // N_HEAD)) / (N_EMBED // N_HEAD) ** 0.5 for t in range(len(kh))]
            aw = softmax(logits)
            head_out = [sum(aw[t] * vh[t][j] for t in range(len(vh))) for j in range(N_EMBED // N_HEAD)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [max(0.0, xi) for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]
    return linear(x, state_dict["lm_head"])


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
    probs = softmax(logits)
    losses.append(-math.log(probs[tokens[pos_id + 1]]))
loss = sum(losses) / n

DATA_OUT.write_text(
    json.dumps(
        {
            "vocab_size": vocab_size,
            "bos": BOS,
            "uchars": "".join(uchars),
            "doc": doc,
            "tokens": tokens,
            "n": n,
            "logits": logits_per_pos,
            "losses": losses,
            "loss": loss,
            "weights": state_dict,
        }
    )
)
print(f"wrote {DATA_OUT} doc={doc!r} n={n} loss={loss:.6f}")
