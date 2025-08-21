from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

from cfm.nn.timestep_embedding import timestep_embedding


class _DotProductAttention(nn.Module):
  n_heads: int = 1

  @nn.compact
  def __call__(self, inputs):
    B, H, W, C = inputs.shape
    chex.assert_equal(C % (3 * self.n_heads), 0)
    q, k, v = jnp.split(inputs, 3, axis=3)
    outputs = nn.attention.dot_product_attention(
      rearrange(q, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
      rearrange(k, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
      rearrange(v, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
    )
    outputs = rearrange(
      outputs,
      "b (h w) heads c -> b h w (heads c)",
      heads=self.n_heads,
      h=H,
      w=W,
    )
    return outputs


class _AttentionBlock(nn.Module):
  n_heads: int
  n_groups: int

  @nn.compact
  def __call__(self, inputs, is_training):
    hidden = inputs
    hidden = nn.GroupNorm(self.n_groups)(hidden)
    # input projection (replacing the MLP on conventional attention)
    hidden = nn.Conv(
      inputs.shape[-1] * 3,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="SAME",
    )(hidden)
    # attention, we don't push through linear layers since we have the
    # convolution above outputting 3 times the layers which we use as
    # k, q, v
    hidden = _DotProductAttention(self.n_heads)(hidden)
    # output projection (replacing the MLP on conventional attention)
    outputs = nn.Conv(
      inputs.shape[-1],
      kernel_size=(1, 1),
      strides=(1, 1),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(hidden)
    return outputs + inputs


class _Downsample(nn.Module):
  use_conv: bool
  kernel_size: int = 3
  stride: int = 2

  @nn.compact
  def __call__(self, inputs, is_training):
    if self.use_conv:
      outputs = nn.Conv(
        inputs.shape[-1],
        kernel_size=(self.kernel_size, self.kernel_size),
        strides=(self.stride, self.stride),
        padding=self.kernel_size // 2,
      )(inputs)
    else:
      outputs = nn.avg_pool(
        inputs,
        window_shape=(self.stride, self.stride),
        strides=(self.stride, self.stride),
      )
    return outputs


class _Upsample(nn.Module):
  use_conv: bool
  n_out_channels: int | None = None
  kernel_size: int = 3
  stride: int = 1

  @nn.compact
  def __call__(self, inputs, is_training):
    B, H, W, C = inputs.shape
    outputs = jax.image.resize(
      inputs,
      (B, H * 2, W * 2, C),
      method="nearest",
    )
    if self.use_conv:
      n_out_channels = self.n_out_channels or inputs.shape[-1]
      outputs = nn.Conv(
        n_out_channels,
        kernel_size=(self.kernel_size, self.kernel_size),
        strides=(self.stride, self.stride),
        padding="SAME",
      )(outputs)
    return outputs


class _ConditionalResidualBlock(nn.Module):
  n_out_channels: int
  dropout_rate: float
  kernel_size: int
  n_groups: int

  @nn.compact
  def __call__(self, inputs, sigma, is_training):
    hidden = inputs
    # convolution with pre-layer norm

    hidden = nn.GroupNorm(num_groups=self.n_groups)(hidden)
    hidden = nn.silu(hidden)
    hidden = nn.Conv(
      self.n_out_channels,
      kernel_size=(self.kernel_size, self.kernel_size),
      strides=(1, 1),
      padding="SAME",
    )(hidden)

    embedding = nn.Dense(self.n_out_channels)(sigma)
    hidden += embedding[:, None, None, :]

    # convolution with pre-layer norm and dropout
    hidden = nn.GroupNorm(num_groups=self.n_groups)(hidden)
    hidden = nn.silu(hidden)
    hidden = nn.Dropout(self.dropout_rate)(
      hidden, deterministic=not is_training
    )
    hidden = nn.Conv(
      self.n_out_channels,
      kernel_size=(self.kernel_size, self.kernel_size),
      strides=(1, 1),
      padding="SAME",
      kernel_init=nn.initializers.zeros,
    )(hidden)

    if inputs.shape[-1] != self.n_out_channels:
      residual = nn.Conv(
        self.n_out_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="SAME",
      )(inputs)
    else:
      residual = inputs

    return hidden + residual


class UNet(nn.Module):
  n_channels: int
  n_out_channels: int
  channel_multipliers: Sequence[int]
  n_resnet_blocks: int
  n_classes: int | None = None
  n_embedding: int = 256
  attention_resolutions: Sequence[int] = ()
  n_attention_heads: int = 2
  kernel_size: int = 3
  dropout_rate: float = 0.1
  use_conv_in_resize: bool = True
  n_groups: int = 32

  @nn.compact
  def __call__(
    self,
    inputs,
    times,
    context=None,
    is_training=False,
    **kwargs,
  ):
    # the input is assumed to be channel last (as is the convention in Flax)
    # B, H, W, C = inputs.shape
    hidden = inputs
    # embed the time points and the conditioning variables
    times = nn.Sequential(
      [
        lambda x: timestep_embedding(times, self.n_embedding),
        nn.Dense(self.n_embedding),
        nn.silu,
        nn.Dense(self.n_embedding),
      ]
    )(times)
    if context is not None and self.n_classes is not None:
      context = nn.Embed(self.n_classes + 1, self.n_embedding)(context)
      times = times + context
    times = nn.silu(times)
    # lift data
    hidden = nn.Conv(
      self.n_channels,
      kernel_size=(self.kernel_size, self.kernel_size),
      strides=(1, 1),
      padding="SAME",
    )(hidden)

    hs = [hidden]
    # downsampling UNet blocks
    for level, channel_mult in enumerate(self.channel_multipliers):
      n_outchannels = channel_mult * self.n_channels
      for _ in range(self.n_resnet_blocks):
        hidden = _ConditionalResidualBlock(
          n_out_channels=n_outchannels,
          dropout_rate=self.dropout_rate,
          kernel_size=self.kernel_size,
          n_groups=self.n_groups,
        )(hidden, times, is_training)
        if hidden.shape[1] in self.attention_resolutions:
          hidden = _AttentionBlock(
            n_heads=self.n_attention_heads, n_groups=self.n_groups
          )(hidden, is_training)
        hs.append(hidden)
      if level != len(self.channel_multipliers) - 1:
        hidden = _Downsample(
          use_conv=self.use_conv_in_resize,
          kernel_size=self.kernel_size,
        )(hidden, is_training)
        hs.append(hidden)

    # middle UNet block
    n_outchannels = self.channel_multipliers[-1] * self.n_channels
    for i in range(2):
      hidden = _ConditionalResidualBlock(
        n_out_channels=n_outchannels,
        dropout_rate=self.dropout_rate,
        kernel_size=self.kernel_size,
        n_groups=self.n_groups,
      )(hidden, times, is_training)
      if i < self.n_resnet_blocks:
        hidden = _AttentionBlock(
          n_heads=self.n_attention_heads, n_groups=self.n_groups
        )(hidden, is_training)

    # upsampling UNet block
    for level, channel_mult in reversed(
      list(enumerate(self.channel_multipliers))
    ):
      n_outchannels = channel_mult * self.n_channels
      for idx in range(self.n_resnet_blocks + 1):
        hidden = jnp.concatenate([hidden, hs.pop()], axis=-1)
        hidden = _ConditionalResidualBlock(
          n_out_channels=n_outchannels,
          dropout_rate=self.dropout_rate,
          kernel_size=self.kernel_size,
          n_groups=self.n_groups,
        )(hidden, times, is_training)
        if hidden.shape[1] in self.attention_resolutions:
          hidden = _AttentionBlock(
            n_heads=self.n_attention_heads, n_groups=self.n_groups
          )(hidden, is_training)
        if level and idx == self.n_resnet_blocks:
          hidden = _Upsample(
            use_conv=self.use_conv_in_resize,
            kernel_size=self.kernel_size,
          )(hidden, is_training)

    outputs = nn.Sequential(
      [
        nn.GroupNorm(self.n_groups),
        nn.silu,
        nn.Conv(
          self.n_out_channels,
          kernel_size=(
            self.kernel_size,
            self.kernel_size,
          ),
          strides=(1, 1),
          padding="SAME",
          kernel_init=nn.initializers.zeros,
        ),
      ]
    )(hidden)
    chex.assert_equal_size([inputs, outputs])
    chex.assert_equal(len(hs), 0)
    return outputs
