"""
v10 checkpoint diagnostics, installed at run time WITHOUT editing the phase modules.

The phase modules (preprocessing, coverage, TNF, TE, encoder) are deliberately left byte-identical: their source hash
is part of every phase / encoder cache key, so editing them would invalidate finished runs. install() wraps, in memory:
  * <module>._fingerprint  -> also registers the hashed parts (utils.checkpoint.register_fp); the hash is unchanged
  * <module>._checkpoint_ok -> before the usual check, logs WHAT changed when a stored fingerprint differs
Checkpoint.mark_done stores the registered parts as "_fp_parts", so the NEXT run can name the changed item.
"""
import importlib

from hyphaesbin.utils.logger import get_logger

log = get_logger("checkpoint")
_MODULES = ("hyphaesbin.preprocessing.preprocessing", "hyphaesbin.coverage.coverage",
            "hyphaesbin.composition.composition.TNF_gene.tnf_gene",
            "hyphaesbin.composition.composition.TE_composition.te_composition",
            "hyphaesbin.encoder.encoder", "hyphaesbin.clustering.clustering")


def _wrap_fp(orig):
    if getattr(orig, "_fp_diag", False):
        return orig

    def _fingerprint(*parts):
        fp = orig(*parts)
        try:
            from hyphaesbin.utils.checkpoint import register_fp
            register_fp(fp, parts)
        except Exception:
            pass
        return fp
    _fingerprint._fp_diag = True
    return _fingerprint


def _wrap_ok(orig):
    if getattr(orig, "_fp_diag", False):
        return orig

    def _checkpoint_ok(ckpt, step, fp, *a, **k):
        try:
            if ckpt is not None and ckpt.is_done(step):
                prev = ckpt.load_metadata(step)
                if prev.get("_fp") != fp:
                    from hyphaesbin.utils.checkpoint import fp_change_reason
                    log.warning(f"{step}: not reused -- CHANGED: {fp_change_reason(prev, fp)}")
        except Exception:
            pass
        return orig(ckpt, step, fp, *a, **k)
    _checkpoint_ok._fp_diag = True
    return _checkpoint_ok


def install() -> None:
    for name in _MODULES:
        try:
            m = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(m, "_fingerprint"):
            m._fingerprint = _wrap_fp(m._fingerprint)
        if hasattr(m, "_checkpoint_ok"):
            m._checkpoint_ok = _wrap_ok(m._checkpoint_ok)
