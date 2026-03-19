import flax.linen as nn
from jax import numpy as jnp

class ResBlockMLP(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x
        x = nn.Dense(self.features)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        
        # skip connection before final ReLU
        return nn.relu(x + residual)

class SimpleResNetMLP(nn.Module):
    policy_head_out_size: int
    num_blocks: int = 2
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True):
        # NOTE: x is BATCHED; first axis is batch dim
        # flatten the input (so that batch axis is the only other axis)
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))

        # shared backbone
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # residual blocks
        for i in range(self.num_blocks):
            x = ResBlockMLP(self.hidden_dim)(x, train=train)

        # policy head -> action logits
        p = nn.Dense(self.hidden_dim // 2)(x)
        p = nn.relu(p)
        policy_logits = nn.Dense(self.policy_head_out_size)(p)

        # value head -> [-1, 1]
        v = nn.Dense(self.hidden_dim // 4)(x)
        v = nn.relu(v)
        value = nn.Dense(1)(v)
        value = jnp.tanh(value)

        return policy_logits, value