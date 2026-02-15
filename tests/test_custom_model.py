import pytest
import torch

from vla.custom_model import CrossAttentionFusion, FlowMatchingActionHead


class TestCrossAttentionFusion:
    @pytest.fixture
    def fusion(self):
        return CrossAttentionFusion(d_model=64, n_heads=4, n_layers=2, dropout=0.0)

    def test_output_shape(self, fusion):
        vision = torch.randn(2, 16, 64)
        lang = torch.randn(2, 8, 64)
        out = fusion(vision, lang)
        assert out.shape == (2, 16, 64)

    def test_different_seq_lengths(self, fusion):
        vision = torch.randn(1, 32, 64)
        lang = torch.randn(1, 5, 64)
        out = fusion(vision, lang)
        assert out.shape == (1, 32, 64)

    def test_gradient_flow(self, fusion):
        vision = torch.randn(2, 16, 64, requires_grad=True)
        lang = torch.randn(2, 8, 64, requires_grad=True)
        out = fusion(vision, lang)
        out.sum().backward()
        assert vision.grad is not None
        assert lang.grad is not None


class TestFlowMatchingActionHead:
    @pytest.fixture
    def head(self):
        return FlowMatchingActionHead(d_model=64, action_dim=7, chunk_size=10, n_layers=2, n_heads=4, dropout=0.0)

    def test_velocity_output_shape(self, head):
        noisy = torch.randn(2, 10, 7)
        t = torch.rand(2)
        cond = torch.randn(2, 16, 64)
        out = head(noisy, t, cond)
        assert out.shape == (2, 10, 7)

    def test_time_broadcast(self, head):
        noisy = torch.randn(4, 10, 7)
        t = torch.rand(4, 1)
        cond = torch.randn(4, 8, 64)
        out = head(noisy, t, cond)
        assert out.shape == (4, 10, 7)

    def test_gradient_flow(self, head):
        noisy = torch.randn(2, 10, 7, requires_grad=True)
        t = torch.rand(2)
        cond = torch.randn(2, 8, 64, requires_grad=True)
        out = head(noisy, t, cond)
        out.sum().backward()
        assert noisy.grad is not None
        assert cond.grad is not None
