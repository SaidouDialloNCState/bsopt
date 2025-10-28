def njit_if_available(signature=None, **kwargs):
    try:
        from numba import njit as _njit
        return _njit(signature, **kwargs)
    except Exception:
        def deco(f): return f
        return deco
