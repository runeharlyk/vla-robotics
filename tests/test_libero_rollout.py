from __future__ import annotations

import sys
import types

from vla.rl.libero_rollout import _patch_robosuite, _safe_close_env


def test_safe_close_env_swallows_keyboard_interrupt() -> None:
    class Env:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            raise KeyboardInterrupt

    env = Env()

    _safe_close_env(env)

    assert env.calls == 1


def test_patch_robosuite_swallows_render_context_cleanup_errors(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "robosuite" or name.startswith("robosuite."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    robosuite_mod = types.ModuleType("robosuite")
    renderers_mod = types.ModuleType("robosuite.renderers")
    context_mod = types.ModuleType("robosuite.renderers.context")
    egl_mod = types.ModuleType("robosuite.renderers.context.egl_context")
    utils_mod = types.ModuleType("robosuite.utils")
    binding_mod = types.ModuleType("robosuite.utils.binding_utils")

    class FakeMjRenderContext:
        def __del__(self) -> None:
            self.del_calls = getattr(self, "del_calls", 0) + 1
            raise AttributeError("broken cleanup")

    class FakeEGLGLContext:
        def free(self) -> None:
            self.free_calls = getattr(self, "free_calls", 0) + 1
            raise RuntimeError("egl cleanup failed")

        def __del__(self) -> None:
            self.del_calls = getattr(self, "del_calls", 0) + 1
            raise RuntimeError("egl del failed")

    binding_mod.MjRenderContext = FakeMjRenderContext
    egl_mod.EGLGLContext = FakeEGLGLContext

    modules = {
        "robosuite": robosuite_mod,
        "robosuite.renderers": renderers_mod,
        "robosuite.renderers.context": context_mod,
        "robosuite.renderers.context.egl_context": egl_mod,
        "robosuite.utils": utils_mod,
        "robosuite.utils.binding_utils": binding_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    _patch_robosuite()

    mj_ctx = FakeMjRenderContext()
    mj_ctx.con = object()
    mj_ctx.__del__()
    assert mj_ctx.del_calls == 1

    mj_ctx_missing = FakeMjRenderContext()
    mj_ctx_missing.__del__()
    assert not hasattr(mj_ctx_missing, "del_calls")

    egl_ctx = FakeEGLGLContext()
    egl_ctx.free()
    egl_ctx.__del__()
    assert egl_ctx.free_calls == 1
    assert egl_ctx.del_calls == 1
