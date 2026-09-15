# Hugging Face SNN → NIR fixtures

Converted `.nir` (HDF5) files produced from **public Hugging Face SNN
checkpoints**, written with the official Python [`nir`](https://pypi.org/project/nir/)
package. They prove `nir-rs` can load graphs that did **not** originate in the
neuromorphs/NIR paper corpus.

CI never talks to Hugging Face. These files are vendored and checksummed.
Synfire registry pulls stay on GitHub [#43](https://github.com/Limen-Neural/nir-rs/issues/43)
/ LIM-1085 and are not mixed into this directory.

## Why convert (not download `.nir`)

Most Hub “SNN” artifacts are framework checkpoints (PyTorch / snnTorch / Norse /
custom), **not** native NIR. NeuroCUDA’s Python `to_nir()` helper returns a
dict for other backends — it does **not** call `nir.write` and is not an HDF5
interchange path. This corpus uses upstream `nir.write` so converter bugs stay
separable from `nir-rs` parser bugs.

## Shortlist (survey)

| Hub repo | License | Format | Export feasibility | Outcome |
|----------|---------|--------|--------------------|---------|
| [`Krishnav1234/neurocuda-mlp-mnist-snn`](https://huggingface.co/Krishnav1234/neurocuda-mlp-mnist-snn) | MIT | `pytorch_model.bin` (fc + IF thresh) | Yes — feed-forward MLP; `nir_exportable: true` | **Vendored** |
| [`Krishnav1234/neurocuda-cnn-nmnist-snn`](https://huggingface.co/Krishnav1234/neurocuda-cnn-nmnist-snn) | MIT | `pytorch_model.bin` (Conv2d + IF + Linear) | Yes — 3-layer CNN; `nir_exportable: true` | **Vendored** |
| [`Krishnav1234/neurocuda-dqn-cartpole-snn`](https://huggingface.co/Krishnav1234/neurocuda-dqn-cartpole-snn) | MIT | `pytorch_model.bin` LIF DQN | Hub marks `nir_exportable: false` | Classified skip |
| [`MIRE-org/flywire-olfactory-snn`](https://huggingface.co/MIRE-org/flywire-olfactory-snn) | MIT | Norse `LIFCell` + custom Poisson / mask loop | NIRTorch cannot trace the time-loop; 800×800 recurrent (~10 MB) | Classified skip |
| [`rmems/Spikenaut-SNN`](https://huggingface.co/rmems/Spikenaut-SNN) | MIT OR Apache-2.0 | Q8.8 `.mem` FPGA bank | No PyTorch graph / NIR exporter | Classified skip |
| [`Catalyst-Neuromorphic/shd-snn-benchmark`](https://huggingface.co/Catalyst-Neuromorphic/shd-snn-benchmark) | MIT | Custom recurrent adLIF | No documented NIR exporter | Classified skip |

Architecture of the two converted models is the NeuroCUDA hub reconstruction
([`neurocuda/hub.py`](https://github.com/Krishnav1/neurocuda/blob/master/neurocuda/hub.py)
`_build_model`), not a one-off parser inside `nir-rs`.

## Conversion toolchain (pinned)

| Tool | Version used |
|------|----------------|
| Python | 3.11 |
| [`nir`](https://pypi.org/project/nir/) | **1.0.8** (`nir.write`, gzip) |
| `torch` | 2.14.0+cpu |
| `huggingface_hub` | 1.31.0 |
| `numpy` | 2.4.6 |
| NeuroCUDA hub architectures | v0.2.0 (`Krishnav1/neurocuda`) |

Not used: NeuroCUDA `to_nir()` dict exporter, Synfire CLI, or any Python step in CI.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'torch>=2.0' --index-url https://download.pytorch.org/whl/cpu
pip install huggingface_hub 'nir==1.0.8' numpy
```

Then run the following (maintainer one-shot; do **not** add this as CI). It
downloads the pinned revisions, builds official NIR graphs, and writes HDF5:

```python
import nir
import numpy as np
import torch
from huggingface_hub import hf_hub_download

def np32(t):
    return t.detach().cpu().numpy().astype(np.float32, copy=False)

def if_node(thresh, shape):
    t = thresh.detach().cpu()
    if t.ndim == 0:
        v_th = np.full(shape, float(t.item()), dtype=np.float32)
    else:
        arr = np32(t)
        v_th = np.broadcast_to(
            arr.reshape((arr.shape[0],) + (1,) * (len(shape) - 1)), shape
        ).copy()
    return nir.IF(r=np.ones(shape, dtype=np.float32), v_threshold=v_th)

# --- MLP MNIST ---------------------------------------------------------------
mlp_rev = "5a2422453d0a1672f5d1dec2ea73d54196a07d85"
mlp = torch.load(
    hf_hub_download(
        "Krishnav1234/neurocuda-mlp-mnist-snn",
        "pytorch_model.bin",
        revision=mlp_rev,
    ),
    map_location="cpu",
    weights_only=True,
)
mlp_graph = nir.NIRGraph(
    nodes={
        "input": nir.Input(input_type={"input": np.array([784], dtype=np.int64)}),
        "fc1": nir.Affine(weight=np32(mlp["fc1.weight"]), bias=np32(mlp["fc1.bias"])),
        "if1": if_node(mlp["relu1.thresh"], (256,)),
        "fc2": nir.Affine(weight=np32(mlp["fc2.weight"]), bias=np32(mlp["fc2.bias"])),
        "if2": if_node(mlp["relu2.thresh"], (256,)),
        "fc3": nir.Affine(weight=np32(mlp["fc3.weight"]), bias=np32(mlp["fc3.bias"])),
        "output": nir.Output(output_type={"output": np.array([10], dtype=np.int64)}),
    },
    edges=[
        ("input", "fc1"),
        ("fc1", "if1"),
        ("if1", "fc2"),
        ("fc2", "if2"),
        ("if2", "fc3"),
        ("fc3", "output"),
    ],
)
nir.write("neurocuda_mlp_mnist.nir", mlp_graph)

# --- CNN N-MNIST -------------------------------------------------------------
cnn_rev = "1ee6ba2f500a4584ef05a77b44016beea2491879"
cnn = torch.load(
    hf_hub_download(
        "Krishnav1234/neurocuda-cnn-nmnist-snn",
        "pytorch_model.bin",
        revision=cnn_rev,
    ),
    map_location="cpu",
    weights_only=True,
)
cnn_graph = nir.NIRGraph(
    nodes={
        "input": nir.Input(input_type={"input": np.array([2, 34, 34], dtype=np.int64)}),
        "conv1": nir.Conv2d(
            input_shape=(34, 34),
            weight=np32(cnn["conv1.weight"]),
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=np32(cnn["conv1.bias"]),
        ),
        "if1": if_node(cnn["act1.thresh"], (32, 34, 34)),
        "pool1": nir.AvgPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        ),
        "conv2": nir.Conv2d(
            input_shape=(17, 17),
            weight=np32(cnn["conv2.weight"]),
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=np32(cnn["conv2.bias"]),
        ),
        "if2": if_node(cnn["act2.thresh"], (64, 17, 17)),
        "pool2": nir.AvgPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        ),
        "conv3": nir.Conv2d(
            input_shape=(8, 8),
            weight=np32(cnn["conv3.weight"]),
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=np32(cnn["conv3.bias"]),
        ),
        "if3": if_node(cnn["act3.thresh"], (128, 8, 8)),
        "pool3": nir.AvgPool2d(
            kernel_size=np.array([2, 2]),
            stride=np.array([2, 2]),
            padding=np.array([0, 0]),
        ),
        "flatten": nir.Flatten(
            input_type={"input": np.array([128, 4, 4], dtype=np.int64)},
            start_dim=0,
            end_dim=-1,
        ),
        "fc": nir.Affine(weight=np32(cnn["fc.weight"]), bias=np32(cnn["fc.bias"])),
        "output": nir.Output(output_type={"output": np.array([10], dtype=np.int64)}),
    },
    edges=[
        ("input", "conv1"),
        ("conv1", "if1"),
        ("if1", "pool1"),
        ("pool1", "conv2"),
        ("conv2", "if2"),
        ("if2", "pool2"),
        ("pool2", "conv3"),
        ("conv3", "if3"),
        ("if3", "pool3"),
        ("pool3", "flatten"),
        ("flatten", "fc"),
        ("fc", "output"),
    ],
)
nir.write("neurocuda_cnn_nmnist.nir", cnn_graph)
```

Verify checksums against [`MANIFEST.toml`](MANIFEST.toml):

```
fc0b1a1e0c4caeb9d1f7be8700de0212a76ec5f441cae13038887411fd9a1ef0  neurocuda_mlp_mnist.nir
972b45984094606b83b5b19173a653524550a5aa416f8a46398baa4250c34f2f  neurocuda_cnn_nmnist.nir
```

Load in `nir-rs` (no Python):

```bash
cargo test --features hdf5 --test hdf5_huggingface
```

## Compatibility notes

- Embedded `/version` is **1.0.8** (the `nir` writer), matching
  `nir_rs::io::DEFAULT_NIR_VERSION`. Paper fixtures remain 0.1.1 / 0.2.0.
- IF `v_reset` is present on the wire (zeros), unlike several paper LIF files
  where it is absent and defaulted on read.
- CNN spatial sizes after `AvgPool2d(2)` on 34×34: 17×17 → 8×8 → 4×4; flatten
  `128×4×4 = 2048` matches `fc.weight` `(10, 2048)`.
- This is a **load/inspect** smoke, not a bit-identical oracle versus the
  original NeuroCUDA forward pass.

## License of the vendored files

The Hugging Face checkpoints are MIT-licensed. The converted `.nir` files
contain those trained weights and are therefore also distributed under MIT
(the NeuroCUDA / Hub license). They do **not** fall under the BSD-3 paper
fixture license in the parent directory.

## Neuromod follow-up

Post-load smoke (map NIR IF / Affine / Conv2d into neuromod primitives) is
tracked separately — see the PR / Linear issue linked from GitHub #44 /
LIM-1086. Implementation stays out of this crate.
