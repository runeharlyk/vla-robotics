from __future__ import annotations

from vla.envs.base import SimEnv, SimEnvFactory

# Lazy registry: maps simulator name → module path + class name.
# The actual import only happens when that specific simulator is requested,
# so missing optional dependencies (e.g. `libero`) won't break unrelated sims.
_REGISTRY: dict[str, tuple[str, str]] = {
    "libero": ("vla.envs.libero", "LiberoEnvFactory"),
    "maniskill": ("vla.envs.maniskill", "ManiSkillEnvFactory"),
}

_CACHE: dict[str, type] = {}


def _get_factory_cls(key: str) -> type:
    if key not in _CACHE:
        module_path, cls_name = _REGISTRY[key]
        import importlib

        mod = importlib.import_module(module_path)
        _CACHE[key] = getattr(mod, cls_name)
    return _CACHE[key]


def make_env_factory(simulator: str, **kwargs) -> SimEnvFactory:
    """Return an env factory for the given simulator backend.

    Args:
        simulator: One of ``"libero"`` or ``"maniskill"``.
        **kwargs: Forwarded to the factory constructor
            (e.g. ``suite`` for LIBERO, ``env_id`` for ManiSkill).

    Only the requested simulator's dependencies are imported, so the other
    simulator's packages don't need to be installed.
    """
    key = simulator.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown simulator {simulator!r}. Available: {sorted(_REGISTRY)}")
    cls = _get_factory_cls(key)
    return cls(**kwargs)


__all__ = ["SimEnv", "SimEnvFactory", "make_env_factory"]
