from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vla.evaluation import visualize
from vla.evaluation.visualize import (
    _parallel_video_name,
    _PreviewWindow,
    _select_parallel_trajectory,
    _trajectory_to_frames,
)


def test_preview_window_noop_when_disabled() -> None:
    preview = _PreviewWindow(enabled=False, fps=30)

    should_stop = preview.show(np.zeros((8, 8, 3), dtype=np.uint8))

    assert should_stop is False


def test_preview_window_returns_true_on_escape() -> None:
    calls: list[tuple[int, int]] = []

    fake_cv2 = SimpleNamespace(
        COLOR_RGB2BGR=0,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0,
        WND_PROP_VISIBLE=0,
        cvtColor=lambda frame, _code: frame,
        resize=lambda frame, size, interpolation=None: frame,
        INTER_LINEAR=0,
        putText=lambda *args, **kwargs: None,
        imshow=lambda *_args, **_kwargs: None,
        waitKey=lambda delay: calls.append((delay, 27)) or 27,
        getWindowProperty=lambda *_args, **_kwargs: 1,
        destroyWindow=lambda *_args, **_kwargs: None,
    )

    with patch.dict(sys.modules, {"cv2": fake_cv2}):
        preview = _PreviewWindow(enabled=True, fps=20)
        should_stop = preview.show(np.zeros((8, 8, 3), dtype=np.uint8), status_lines=("task",))
        preview.close()

    assert should_stop is True
    assert calls == [(50, 27)]


def test_visualize_uses_inference_mode_and_autocast(monkeypatch) -> None:
    captured: dict[str, list[dict[str, object]]] = {"autocast": [], "inference_mode": []}

    class _FakePolicy:
        def eval(self):
            return self

        def reset(self) -> None:
            pass

        def select_action(self, batch: dict) -> torch.Tensor:
            return torch.zeros((1, 7), dtype=torch.float32)

    class _FakeEnv:
        task_description = "fake task"
        max_episode_steps = 1

        def reset(self, seed: int = 0):
            return {"pixels": {"image": np.zeros((8, 8, 3), dtype=np.uint8)}}, {}

        def obs_to_batch(self, raw_obs: dict, device=None) -> dict:
            image = torch.zeros((1, 3, 8, 8), dtype=torch.float32, device=device)
            return {"observation.images.image": image, "task": [self.task_description]}

        def step(self, action: np.ndarray):
            obs = {"pixels": {"image": np.zeros((8, 8, 3), dtype=np.uint8)}}
            return obs, 0.0, True, False, {}

        def get_frame(self, raw_obs: dict) -> np.ndarray:
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def is_success(self, info: dict) -> bool:
            return False

        def close(self) -> None:
            pass

    class _FakeFactory:
        suite_name = "fake"
        num_tasks = 1

        def __call__(self, task_id: int) -> _FakeEnv:
            return _FakeEnv()

    @contextmanager
    def _fake_inference_mode():
        captured["inference_mode"].append({})
        yield

    @contextmanager
    def _fake_autocast(device_type: str, dtype=None, enabled: bool = False):
        captured["autocast"].append(
            {
                "device_type": device_type,
                "dtype": dtype,
                "enabled": enabled,
            }
        )
        yield

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.inference_mode", _fake_inference_mode)
    monkeypatch.setattr("torch.autocast", _fake_autocast)
    monkeypatch.setattr(
        "vla.models.load_policy",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy=_FakePolicy(),
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            state_dim=0,
            action_dim=7,
        ),
    )
    monkeypatch.setattr("vla.evaluation.evaluate._make_factory", lambda *_args, **_kwargs: _FakeFactory())
    monkeypatch.setattr(visualize, "_save_video", lambda *_args, **_kwargs: None)

    visualize.main(
        model="smolvla",
        checkpoint="fake-checkpoint",
        simulator="libero",
        suite="fake",
        env_id=None,
        episodes=1,
        device="cuda",
        output_dir="videos-test",
        tasks=None,
        fps=30,
        seed=0,
        max_steps=None,
        num_envs=1,
        show=False,
        preview_scale=2.0,
    )

    assert len(captured["inference_mode"]) == 1
    assert captured["autocast"] == [
        {
            "device_type": "cuda",
            "dtype": torch.bfloat16,
            "enabled": True,
        }
    ]


def test_trajectory_to_frames_concatenates_views() -> None:
    trajectory = SimpleNamespace(
        images=torch.tensor(
            [
                [
                    [[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]],
                    [[[4, 4], [4, 4]], [[5, 5], [5, 5]], [[6, 6], [6, 6]]],
                ]
            ],
            dtype=torch.uint8,
        )
    )

    frames = _trajectory_to_frames(trajectory)

    assert len(frames) == 1
    assert frames[0].shape == (2, 4, 3)
    assert np.array_equal(frames[0][:, :2], np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]], dtype=np.uint8))
    assert np.array_equal(frames[0][:, 2:], np.array([[[4, 5, 6], [4, 5, 6]], [[4, 5, 6], [4, 5, 6]]], dtype=np.uint8))


def test_select_parallel_trajectory_prefers_first_success() -> None:
    trajectories = [
        SimpleNamespace(success=False),
        SimpleNamespace(success=True),
        SimpleNamespace(success=True),
    ]

    idx, selected = _select_parallel_trajectory(trajectories)

    assert idx == 1
    assert selected is trajectories[1]


def test_parallel_video_name_includes_attempt_and_status() -> None:
    assert _parallel_video_name(3, 7, 2, True) == "task03_ep07_try02_success.mp4"
    assert _parallel_video_name(3, 7, 5, False) == "task03_ep07_try05_fail.mp4"


def test_visualize_respects_max_steps_override(monkeypatch) -> None:
    step_count = 0

    class _FakePolicy:
        def eval(self):
            return self

        def reset(self) -> None:
            pass

        def select_action(self, batch: dict) -> torch.Tensor:
            return torch.zeros((1, 7), dtype=torch.float32)

    class _FakeEnv:
        task_description = "fake task"
        max_episode_steps = 5

        def reset(self, seed: int = 0):
            return {"pixels": {"image": np.zeros((8, 8, 3), dtype=np.uint8)}}, {}

        def obs_to_batch(self, raw_obs: dict, device=None) -> dict:
            image = torch.zeros((1, 3, 8, 8), dtype=torch.float32, device=device)
            return {"observation.images.image": image, "task": [self.task_description]}

        def step(self, action: np.ndarray):
            nonlocal step_count
            step_count += 1
            obs = {"pixels": {"image": np.zeros((8, 8, 3), dtype=np.uint8)}}
            return obs, 0.0, False, False, {}

        def get_frame(self, raw_obs: dict) -> np.ndarray:
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def is_success(self, info: dict) -> bool:
            return False

        def close(self) -> None:
            pass

    class _FakeFactory:
        suite_name = "fake"
        num_tasks = 1

        def __call__(self, task_id: int) -> _FakeEnv:
            return _FakeEnv()

    monkeypatch.setattr(
        "vla.models.load_policy",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy=_FakePolicy(),
            preprocessor=lambda batch: batch,
            postprocessor=lambda action: action,
            state_dim=0,
            action_dim=7,
        ),
    )
    monkeypatch.setattr("vla.evaluation.evaluate._make_factory", lambda *_args, **_kwargs: _FakeFactory())
    monkeypatch.setattr(visualize, "_save_video", lambda *_args, **_kwargs: None)

    visualize.main(
        model="smolvla",
        checkpoint="fake-checkpoint",
        simulator="libero",
        suite="fake",
        env_id=None,
        episodes=1,
        device="cpu",
        output_dir="videos-test",
        tasks=None,
        fps=30,
        seed=0,
        max_steps=2,
        num_envs=1,
        show=False,
        preview_scale=2.0,
    )

    assert step_count == 2
