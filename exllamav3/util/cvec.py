"""
Residual-stream control vectors (research): project a direction out of the stream after every block, or record
the stream for measuring such directions.

EXL3_CVEC        .safetensors with tensors "direction.N": one unit vector of the hidden width per block index N,
                 or a [k, hidden] matrix of k orthonormal directions (blocks without a tensor are left alone)
EXL3_CVEC_SCALE  strength, default 1.0: after block N, x -= scale * (x . v_N) v_N on every hyper-connection
                 stream. 1.0 removes the component, the weights are untouched (the same effect as orthogonalizing
                 every matrix that writes into the stream, without requantizing anything)

The hook sits in the module walk of both forward paths (layer split and tensor parallel). Every tensor-parallel
rank applies the same operation to the same stream, so the ranks stay bit-identical.

Recording (in-process tools, main process only): capture_begin() ... forward ... capture_end() returns the stream at
the last position after every block, [blocks, batch, streams, hidden].
"""
import os
import re
import torch

_FILE = os.environ.get("EXL3_CVEC", "")
_SCALE = float(os.environ.get("EXL3_CVEC_SCALE", "1.0"))
ENABLED = bool(_FILE)
ACTIVE = ENABLED              # projection or recording: the one flag the forward loops test

_vectors = None               # block index -> host tensor
_on_device = {}               # (device, block index, dtype) -> tensor
_capture = None
_BLOCK = re.compile(r"layers\.(\d+)$")


def _block_index(module):
    idx = getattr(module, "_cvec_block", None)
    if idx is None:
        m = _BLOCK.search(getattr(module, "key", "") or "")
        idx = int(m.group(1)) if m else -1
        module._cvec_block = idx
    return idx


def _vector(idx, x):
    global _vectors
    if _vectors is None:
        from safetensors.torch import load_file
        _vectors = {int(k.split(".")[1]): v.float() for k, v in load_file(_FILE).items() if k.startswith("direction.")}
        print(f" -- cvec: {len(_vectors)} directions from {_FILE}, scale {_SCALE}", flush = True)
    v = _vectors.get(idx)
    if v is None:
        return None
    key = (x.device, idx, x.dtype)
    t = _on_device.get(key)
    if t is None:
        t = _on_device[key] = v.to(x.device, x.dtype)
    return t


_hc_flush = None


def after_module(module, x, params = None):
    """Called with a module's output: records and / or projects the stream after a transformer block"""
    global _hc_flush
    idx = _block_index(module)
    if idx < 0 or x is None or not x.is_floating_point():
        return x
    v = _vector(idx, x) if ENABLED else None
    if v is not None and x.shape[-1] != v.shape[-1]:
        v = None
    if (v is not None or _capture is not None) and params is not None and "hc_pending" in params:
        # A decode-sized block leaves its last stream update pending for the next block's mix (EXL3_HC_FOLD).
        # The stream is read here, so it has to be complete first
        if _hc_flush is None:
            from ..modules.hyperconnections import hc_flush
            _hc_flush = hc_flush
        _hc_flush(params)
    if _capture is not None:
        _capture.append(x[:, -1].detach().float().clone())
    if v is not None:
        if v.dim() == 1:
            c = torch.matmul(x, v)
            x.addcmul_(c.unsqueeze(-1), v, value = -_SCALE)
        else:
            # several orthonormal directions for this block, rows of v: x -= scale * (x V^T) V
            x.sub_(torch.matmul(torch.matmul(x, v.transpose(0, 1)), v), alpha = _SCALE)
    return x


def capture_begin():
    global _capture, ACTIVE
    _capture = []
    ACTIVE = True


def capture_end():
    """[blocks, batch, streams, hidden] (streams = 1 for a plain residual stream) on the host"""
    global _capture, ACTIVE
    out = _capture
    _capture = None
    ACTIVE = ENABLED
    if not out:
        return None
    t = torch.stack([(o if o.dim() == 3 else o.unsqueeze(1)).cpu() for o in out])    # blocks may sit on different devices
    return t


def configure(file = "", scale = 1.0):
    """Switch the control vector inside this process (layer-split tools; tensor-parallel ranks read the environment)"""
    global _FILE, _SCALE, ENABLED, ACTIVE, _vectors
    _FILE, _SCALE = file or "", float(scale)
    ENABLED = bool(_FILE)
    ACTIVE = ENABLED or _capture is not None
    _vectors = None
    _on_device.clear()
