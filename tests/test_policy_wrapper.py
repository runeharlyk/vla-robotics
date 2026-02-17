import pytest
import torch
import torch.nn as nn

from vla.models.policy_wrapper import ActionPolicy, PolicyWrapper


class DummyPolicy(nn.Module):
    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.linear = nn.Linear(3, action_dim)
        self._action_dim = action_dim

    def select_action(self, batch: dict) -> torch.Tensor:
        return torch.zeros(1, self._action_dim)

    def reset(self) -> None:
        pass

    def forward(self, batch: dict) -> dict:
        return {"loss": torch.tensor(0.0)}


class TestActionPolicy:
    def test_protocol_check(self):
        p = DummyPolicy()
        assert isinstance(p, ActionPolicy)


class TestPolicyWrapper:
    @pytest.fixture
    def wrapper(self):
        return PolicyWrapper(DummyPolicy(action_dim=7), device="cpu")

    def test_device_property(self, wrapper):
        assert wrapper.device == torch.device("cpu")

    def test_select_action(self, wrapper):
        batch = {"images": torch.randn(1, 3, 64, 64)}
        action = wrapper.select_action(batch)
        assert action.shape == (1, 7)

    def test_reset(self, wrapper):
        wrapper.reset()

    def test_forward(self, wrapper):
        batch = {"images": torch.randn(1, 3, 64, 64)}
        out = wrapper(batch)
        assert "loss" in out
