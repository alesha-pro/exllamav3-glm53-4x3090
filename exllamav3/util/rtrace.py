"""
Router trace (research): the router logits of every MoE layer for every forward pass, recorded in the main
process only. Used to study expert reuse between tokens and what a prefetcher of cold experts could know ahead.

    rtrace.begin(leads = ()) ... generate ... passes, preds = rtrace.end()
    passes: list of [layers, rows, experts] fp16 host tensors, one per forward pass, layers in module order
    preds:  with leads (e.g. (1, 2, 3)): for single-row passes, [leads, layers, experts] fp16 per pass: the logits
            layer L's router would give if it were run on the stream as it left the block `lead` blocks before L
            (its own stream mix and norm applied to that earlier stream), i.e. what could be known that much
            earlier in the same token. -30000 where there is no such earlier block. Run with EXL3_HC_FOLD=0.
"""
import torch

ACTIVE = False
LEADS = ()
_passes = []
_preds = []
_cur = None
_keys = None
_pcur = None
_first = None


def begin(leads = ()):
    global ACTIVE, LEADS, _passes, _preds, _cur, _keys, _pcur, _first
    _passes, _preds, _cur, _keys, _pcur, _first = [], [], None, None, None, None
    LEADS = tuple(leads)
    ACTIVE = True


def rec(module, bsz, cfg, device):
    """Called by BlockSparseMLP right after it routed: copy that call's router logits"""
    global _cur, _keys, _pcur, _first
    from ..modules.block_sparse_mlp_routing import _routing_buffers
    if _first is None:
        _first = module.key
    if module.key == _first:
        _flush()
        _cur, _keys, _pcur = [], [], {}
    logits = cfg.router_logits_bsz1 if bsz == 1 else _routing_buffers(cfg, bsz, device)[0]
    _cur.append(logits[:bsz].detach().clone())
    _keys.append(module.key)


def predict(modules, idx, x):
    """Called from the module walk with the stream after block idx (single-row passes): run the routers of the
    blocks LEADS further on against this stream"""
    if _pcur is None or x is None or x.dim() != 4 or x.shape[0] * x.shape[1] != 1:
        return
    for lead in LEADS:
        j = idx + lead
        if j >= len(modules):
            continue
        tm = modules[j]
        mlp = getattr(tm, "mlp", None)
        cfg = getattr(mlp, "routing_cfg", None) if mlp is not None else None
        hc = getattr(tm, "mlp_hc", None)
        if cfg is None or hc is None:
            continue
        p2 = {}
        y = hc.mix_norm(x, p2, None)[2].clone()
        if tm.mlp_norm is not None:
            y = tm.mlp_norm.forward(y, p2, out_dtype = torch.half).clone()
        if mlp.router_pre_norm:
            y = mlp.router_pre_norm.forward(y, p2, out_dtype = torch.half).clone()
        g = cfg.gate_tensor
        yf = y.reshape(1, -1).float()
        logits = torch.matmul(yf, g.float()) if g.shape[0] == yf.shape[1] else torch.matmul(yf, g.float().T)
        _pcur[(lead, mlp.key)] = logits[0].half()


def _flush():
    global _cur
    if _cur:
        _passes.append(torch.stack(_cur).cpu())
        if LEADS:
            if _cur[0].shape[0] == 1:
                E = _cur[0].shape[-1]
                p = torch.full((len(LEADS), len(_keys), E), -30000.0, dtype = torch.half)
                for (lead, key), v in _pcur.items():
                    if key in _keys:
                        p[LEADS.index(lead), _keys.index(key)] = v.cpu()
                _preds.append(p)
            else:
                _preds.append(None)
    _cur = None


def end():
    global ACTIVE, LEADS
    _flush()
    ACTIVE = False
    LEADS = ()
    return _passes, _preds
