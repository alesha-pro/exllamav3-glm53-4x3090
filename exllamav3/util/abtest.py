"""
A/B switch helper (debug): set a module-level attribute in this process and, for a
tensor-parallel model, in every rank's worker process, so one loaded model can be measured under
several settings of a Python-side switch.

    from exllamav3.util import abtest
    abtest.set_global(model, "exllamav3.modules.hyperconnections", "_hc_fold", True)
"""
import importlib


def mp_setattr(local_context, modname: str, attr: str, value):
    m = importlib.import_module(modname)
    setattr(m, attr, value)
    return True


def set_global(model, modname: str, attr: str, value):
    mp_setattr(None, modname, attr, value)
    for mdl in (model if isinstance(model, (list, tuple)) else [model]):
        if mdl is not None and getattr(mdl, "loaded_tp", False):
            mdl.tp_dispatch_all(mp_setattr, (modname, attr, value))


def mp_backend_setattr(local_context, attr: str, value):
    setattr(local_context["backend"], attr, value)
    return True


def set_backend(model, attr: str, value):
    """Set an attribute of every rank's TP backend object (e.g. p2p = False for an A/B pass)"""
    for mdl in (model if isinstance(model, (list, tuple)) else [model]):
        if mdl is not None and getattr(mdl, "loaded_tp", False):
            mdl.tp_dispatch_all(mp_backend_setattr, (attr, value))


def mp_call(local_context, modname: str, fn: str, args: tuple):
    return getattr(importlib.import_module(modname), fn)(*args)


def call_all(model, modname: str, fn: str, *args):
    """Call a module-level function in this process and in every TP rank's worker"""
    done = False
    for mdl in (model if isinstance(model, (list, tuple)) else [model]):
        if mdl is not None and getattr(mdl, "loaded_tp", False):
            mdl.tp_dispatch_all(mp_call, (modname, fn, args))   # includes the in-process output rank
            done = True
    if not done:
        mp_call(None, modname, fn, args)
