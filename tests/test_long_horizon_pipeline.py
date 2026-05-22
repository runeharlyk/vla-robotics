from __future__ import annotations

import numpy as np

from vla.planning.long_horizon_pipeline import (
    JudgeDecision,
    extract_camera_image,
    run_orchestrated_episode,
)
from vla.planning.smolvlm_decompose import DecompositionResult, Subgoal


class FakeEnv:
    def __init__(self, success_after: int = 2) -> None:
        self.max_episode_steps = 10
        self._step = 0
        self._success_after = success_after

    @property
    def task_description(self) -> str:
        return "open the drawer and place the bowl on the plate"

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        self._step = 0
        return self._make_obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._step += 1
        info = {"is_success": self._step >= self._success_after}
        return self._make_obs(), 0.0, False, False, info

    def close(self) -> None:
        return None

    def obs_to_batch(self, raw_obs: dict, device=None) -> dict:
        return {"task": [self.task_description]}

    def get_frame(self, raw_obs: dict) -> np.ndarray:
        return next(iter(raw_obs["pixels"].values()))

    def is_success(self, info: dict) -> bool:
        return bool(info.get("is_success", False))

    def _make_obs(self) -> dict:
        frame = np.full((8, 8, 3), fill_value=min(self._step, 255), dtype=np.uint8)
        return {
            "pixels": {
                "agentview_image": frame,
                "robot0_eye_in_hand_image": frame,
            }
        }


class FakeExecutor:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def reset(self) -> None:
        return None

    def act(self, *, env, raw_obs: dict, instruction: str) -> np.ndarray:
        self.instructions.append(instruction)
        return np.zeros(7, dtype=np.float32)


class FakePlanner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def plan(
        self,
        *,
        task_instruction: str,
        image,
        completed_subgoals: list[str],
        max_subgoals: int,
    ) -> DecompositionResult:
        self.calls.append(list(completed_subgoals))
        if not completed_subgoals:
            subgoals = [
                Subgoal(1, "Open drawer", "open the drawer", ""),
                Subgoal(2, "Move bowl", "place the bowl on the plate", ""),
            ]
        else:
            subgoals = [Subgoal(1, "Move bowl", "place the bowl on the plate", "")]
        return DecompositionResult(
            instruction=task_instruction,
            model_id="fake-smolvlm",
            task_summary=task_instruction,
            subgoals=subgoals,
            raw_response="fake",
        )


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(
        self,
        *,
        task_instruction: str,
        active_subgoal: str,
        image,
        completed_subgoals: list[str],
        threshold: float,
    ) -> JudgeDecision:
        self.calls += 1
        return JudgeDecision(
            predicted_label="Yes",
            yes_prob=0.97,
            no_prob=0.03,
            threshold=threshold,
            solved=True,
            prompt="fake",
        )


def test_extract_camera_image_prefers_requested_camera() -> None:
    raw_obs = {
        "pixels": {
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        }
    }

    image = extract_camera_image(raw_obs, camera_name="agentview_image", image_size=16)

    assert image is not None
    assert image.size == (16, 16)


def test_run_orchestrated_episode_replans_after_solved_subgoal() -> None:
    env = FakeEnv(success_after=2)
    executor = FakeExecutor()
    planner = FakePlanner()
    judge = FakeJudge()

    result = run_orchestrated_episode(
        env=env,
        task_instruction=env.task_description,
        executor=executor,
        planner=planner,
        judge=judge,
        seed=0,
        max_steps=4,
        max_subgoals=4,
        judge_interval_seconds=1.0,
        control_hz=1.0,
        yes_threshold=0.9,
        replan_on_completion=True,
        visual_token_estimate=1024,
    )

    assert result.success is True
    assert result.visual_token_estimate == 1024
    assert result.solved_subgoals == ["open the drawer", "place the bowl on the plate"]
    assert planner.calls == [[], ["open the drawer"]]
    assert judge.calls == 1
    assert len(result.plan_events) == 2
    assert executor.instructions[0] == "open the drawer"


def test_run_orchestrated_episode_stops_when_no_subgoals_remain() -> None:
    env = FakeEnv(success_after=99)
    executor = FakeExecutor()

    class OneShotPlanner(FakePlanner):
        def plan(self, *, task_instruction: str, image, completed_subgoals: list[str], max_subgoals: int):
            if completed_subgoals:
                subgoals: list[Subgoal] = []
            else:
                subgoals = [Subgoal(1, "Open drawer", "open the drawer", "")]
            return DecompositionResult(
                instruction=task_instruction,
                model_id="fake-smolvlm",
                task_summary=task_instruction,
                subgoals=subgoals,
                raw_response="fake",
            )

    planner = OneShotPlanner()
    judge = FakeJudge()
    result = run_orchestrated_episode(
        env=env,
        task_instruction=env.task_description,
        executor=executor,
        planner=planner,
        judge=judge,
        seed=0,
        max_steps=3,
        max_subgoals=4,
        judge_interval_seconds=1.0,
        control_hz=1.0,
        yes_threshold=0.9,
        replan_on_completion=True,
    )

    assert result.success is False
    assert result.solved_subgoals == ["open the drawer"]
    assert result.remaining_subgoals == []
