"""State-based action model for robot manipulation.

This module uses proprioceptive state (joint positions, velocities, etc.)
to predict robot actions, as a fallback when RGB observations are unavailable.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class StateActionConfig:
    """Configuration for state-based action model.

    Args:
        state_dim: Dimension of the state observation.
        action_dim: Dimension of the action space.
        hidden_dim: Hidden dimension for MLP layers.
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
    """

    state_dim: int = 32
    action_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1


class StateActionModel(nn.Module):
    """MLP model for predicting robot actions from state observations."""

    def __init__(self, config: StateActionConfig):
        """Initialize the state action model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        current_dim = config.state_dim

        for i in range(config.num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
            current_dim = config.hidden_dim

        layers.append(nn.Linear(current_dim, config.action_dim))
        self.mlp = nn.Sequential(*layers)

        self.action_scale = nn.Parameter(torch.ones(config.action_dim))
        self.action_bias = nn.Parameter(torch.zeros(config.action_dim))

        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict actions.

        Args:
            state: Input state of shape (batch_size, state_dim).

        Returns:
            Predicted actions of shape (batch_size, action_dim).
        """
        state = state.to(self.device)
        raw_actions = self.mlp(state)
        actions = raw_actions * self.action_scale + self.action_bias
        return actions

    def compute_loss(
        self,
        state: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss.

        Args:
            state: Input state of shape (batch_size, state_dim).
            target_actions: Ground truth actions of shape (batch_size, action_dim).

        Returns:
            Dictionary with 'total' loss and component losses.
        """
        predicted_actions = self.forward(state)
        target_actions = target_actions.to(self.device)

        mse_loss = nn.functional.mse_loss(predicted_actions, target_actions)
        l1_loss = nn.functional.l1_loss(predicted_actions, target_actions)

        total_loss = mse_loss + 0.1 * l1_loss

        return {
            "total": total_loss,
            "mse": mse_loss,
            "l1": l1_loss,
        }


def create_state_model(state_dim: int = 32, action_dim: int = 8) -> StateActionModel:
    """Create a state-based action model.

    Args:
        state_dim: Dimension of state observations.
        action_dim: Dimension of action space.

    Returns:
        Initialized StateActionModel.
    """
    config = StateActionConfig(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    return StateActionModel(config)
