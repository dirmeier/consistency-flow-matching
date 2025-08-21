import chex
import jax
import numpy as np
from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp

from cfm.nn.timestep_embedding import timestep_embedding


def _modulate(inputs, shift, scale):
  return inputs * (1.0 + scale[:, None]) + shift[:, None]


def get_sinusoidal_embedding_1d(length, embedding_dim):
  emb = timestep_embedding(length.reshape(-1), embedding_dim)
  return emb


def sinusoidal_init(rng, shape, dtype):
  def get_sinusoidal_embedding_2d(grid, embedding_dim):
    emb_h = get_sinusoidal_embedding_1d(grid[0], embedding_dim // 2)
    emb_w = get_sinusoidal_embedding_1d(grid[1], embedding_dim // 2)
    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb

  _, length, embedding_dim = shape
  grid_size = int(length**0.5)

  grid_h = jnp.arange(grid_size, dtype=jnp.float32)
  grid_w = jnp.arange(grid_size, dtype=jnp.float32)
  grid = jnp.meshgrid(grid_w, grid_h)
  grid = jnp.stack(grid, axis=0)
  grid = grid.reshape([2, 1, grid_size, grid_size])
  pos_embed = get_sinusoidal_embedding_2d(grid, embedding_dim)
  return jnp.expand_dims(pos_embed, 0)  # (1, H*W, D)


class DiTBlock(nn.Module):
  hidden_size: int
  n_heads: int
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, context, is_training, **kwargs):
    hidden = inputs
    adaln_norm = nn.Dense(self.hidden_size * 6)(context)
    attn, gate = jnp.split(adaln_norm, 2, axis=-1)

    pre_shift, pre_scale, post_scale = jnp.split(attn, 3, -1)
    intermediate = nn.LayerNorm(use_scale=False, use_bias=False)(hidden)
    intermediate = _modulate(intermediate, pre_shift, pre_scale)
    intermediate = nn.SelfAttention(num_heads=self.n_heads)(intermediate)
    hidden = hidden + post_scale[:, None] * intermediate

    pre_shift, pre_scale, post_scale = jnp.split(gate, 3, -1)
    intermediate = nn.LayerNorm(use_scale=False, use_bias=False)(hidden)
    intermediate = _modulate(intermediate, pre_shift, pre_scale)
    intermediate = nn.Sequential(
      [
        nn.Dense(self.hidden_size * 4),
        nn.gelu,
        lambda x: nn.Dropout(self.dropout_rate)(
          x, deterministic=not is_training
        ),
        nn.Dense(self.hidden_size),
        lambda x: nn.Dropout(self.dropout_rate)(
          x, deterministic=not is_training
        ),
      ]
    )(intermediate)
    outputs = hidden + post_scale[:, None] * intermediate
    return outputs


class DiT(nn.Module):
  n_channels: int
  n_out_channels: int
  patch_size: int
  n_blocks: int
  n_heads: int
  dropout_rate: float = 0.1
  n_frequency_embedding_size: int = 256
  n_classes: int | None = None

  def _time_embedding(self, times):
    times = timestep_embedding(times, self.n_frequency_embedding_size)
    times = nn.Sequential(
      [
        nn.Dense(self.n_frequency_embedding_size),
        nn.swish,
        nn.Dense(self.n_channels),
      ]
    )(times)
    return times

  def _patchify(self, inputs):
    B, H, W, C = inputs.shape
    patch_size_tuple = (self.patch_size, self.patch_size)
    n_patches = H // self.patch_size
    hidden = nn.Conv(
      self.n_channels,
      patch_size_tuple,
      patch_size_tuple,
      padding="VALID",
      kernel_init=nn.initializers.xavier_uniform(),
    )(inputs)
    outputs = rearrange(
      hidden, "b h w c -> b (h w) c", h=n_patches, w=n_patches
    )
    return outputs

  def _unpatchify(self, inputs):
    B, HW, *_ = inputs.shape
    h = w = int(np.sqrt(HW))
    p = q = self.patch_size
    hidden = jnp.reshape(
      inputs,
      (B, h, w, p, q, self.n_out_channels),
    )
    outputs = rearrange(
      hidden, "b h w p q c -> b (h p) (w q) c", h=h, w=w, p=q, q=q
    )
    return outputs

  def _embed(self, inputs):
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    patch_embedding = self.param(
      "patch_embedding",
      sinusoidal_init,
      pos_emb_shape,
      inputs.dtype,
    )
    patch_embedding = jax.lax.stop_gradient(patch_embedding)
    return inputs + patch_embedding

  @nn.compact
  def __call__(self, inputs, times, context=None, is_training=True):
    hidden = self._patchify(inputs)
    hidden = self._embed(hidden)
    times = self._time_embedding(times)
    if context is not None and self.n_classes is not None:
      context = nn.Embed(self.n_classes + 1, self.n_embedding)(context)
      times = times + context
    times = nn.swish(times)

    for i in range(self.n_blocks):
      hidden = DiTBlock(self.n_channels, self.n_heads, self.dropout_rate)(
        hidden, context=times, is_training=is_training
      )

    # final layer
    times = nn.Dense(self.n_channels * 2, kernel_init=nn.initializers.zeros)(
      times
    )
    times_shift, times_scale = jnp.split(times, 2, -1)
    hidden = nn.Sequential(
      [
        nn.LayerNorm(use_scale=False, use_bias=False),
        lambda x: _modulate(x, times_shift, times_scale),
        nn.Dense(
          self.patch_size * self.patch_size * self.n_out_channels,
          kernel_init=nn.initializers.zeros,
        ),
      ]
    )(hidden)
    outputs = self._unpatchify(hidden)
    chex.assert_equal_size([inputs, outputs])
    return outputs
