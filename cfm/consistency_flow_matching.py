import dataclasses
from collections.abc import Callable

import chex
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import random as jr


def _f_euler(times_t, times_seq, inputs_t, vt):
  new_shape = (-1,) + tuple(np.ones(inputs_t.ndim - 1, dtype=np.int32).tolist())
  times_diff = (times_seq - times_t).reshape(new_shape)
  eul = inputs_t + times_diff * vt
  return eul


def _threshold_based_f_euler(
  times_t, times_seq, inputs_t, vt, inputs_seq, threshold
):
  new_shape = (-1,) + tuple(np.ones(inputs_t.ndim - 1, dtype=np.int32).tolist())
  less_than_threshold = (
    (times_t < threshold).astype(jnp.float32).reshape(new_shape)
  )
  not_less_than_threshold = (
    (times_t >= threshold).astype(jnp.float32).reshape(new_shape)
  )
  ret = less_than_threshold * _f_euler(times_t, times_seq, inputs_t, vt)
  ret = ret + not_less_than_threshold * inputs_seq
  return ret


def _forward_process(inputs, times, noise):
  new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
  times = times.reshape(new_shape)
  inputs_t = times * inputs + (1.0 - times) * noise
  return inputs_t


@dataclasses.dataclass
class ConsistencyFlowMatchingConfig:
  n_sampling_steps: int = 10
  delta: float = 1e-3
  alpha: float = 1e-5
  time_eps: float = 1e-3
  time_max: float = 1.0
  num_segments: int = 2
  threshold: float = 0.9


class ConsistencyFlowMatching:
  def __new__(cls, model_fn: Callable, config: ConsistencyFlowMatchingConfig):
    def loss_fn(params, rng_key, inputs, is_training, context=None, **kwargs):
      time_key, rng_key = jr.split(rng_key)
      times_t = jr.uniform(time_key, shape=(inputs.shape[0],))
      times_t = times_t * (config.time_max - config.time_eps) + config.time_eps
      times_r = jnp.clip(times_t + config.delta, a_max=config.time_max)
      segments = jnp.linspace(
        config.time_eps, config.time_max, config.num_segments + 1
      )
      seg_indices = jnp.searchsorted(segments, times_t, side="left").clip(min=1)
      times_seg = segments[seg_indices]

      noise_key, rng_key = jr.split(rng_key)
      noise = jr.normal(noise_key, inputs.shape)
      inputs_t = _forward_process(inputs, times_t, noise)
      inputs_r = _forward_process(inputs, times_r, noise)
      inputs_seg = _forward_process(inputs, times_seg, noise)

      dropout_key1, dropout_key2, rng_key = jr.split(rng_key, 3)
      vt = model_fn(
        variables={"params": params},
        rngs={"dropout": dropout_key1},
        inputs=inputs_t,
        times=times_t * 999.0,
        context=context,
        is_training=is_training,
      )
      vr = model_fn(
        variables={"params": params},
        rngs={"dropout": dropout_key2},
        inputs=inputs_r,
        times=times_r * 999.0,
        context=context,
        is_training=is_training,
      )
      vr = lax.stop_gradient(vr)

      ft = _f_euler(times_t, times_seg, inputs_t, vt)
      fr = _threshold_based_f_euler(
        times_r, times_seg, inputs_r, vr, inputs_seg, config.threshold
      )
      loss_f = jnp.mean(jnp.square(ft - fr), axis=range(1, inputs.ndim))

      far_from_segment_ends = (times_seg - times_t) > 1.01 * config.delta
      thresh = (times_t < config.threshold).astype(
        jnp.float32
      ) * far_from_segment_ends
      loss_v = jnp.square(vt - vr)
      loss_v = thresh * jnp.mean(loss_v, axis=range(1, inputs.ndim))

      loss = loss_f + config.alpha * loss_v
      return loss.mean()

    def sample_fn(
      rng_key,
      state,
      sample_shape=(),
      context=None,
      **kwargs,
    ):
      """Conventional discretized Euler sampler."""
      if context is not None:
        chex.assert_equal(sample_shape[0], len(context))
      dt = 1.0 / config.n_sampling_steps
      y = jr.normal(rng_key, sample_shape)
      for i in range(config.n_sampling_steps):
        times = i / config.n_sampling_steps
        times = times * (config.time_max - config.time_eps) + config.time_eps
        times = jnp.repeat(times, y.shape[0])
        vt = state.apply_fn(
          variables={"params": state.ema_params},
          inputs=y,
          times=times * 999.0,
          context=context,
          is_training=False,
        )
        y = y + vt * dt
      return y

    return loss_fn, lambda x: x, sample_fn
