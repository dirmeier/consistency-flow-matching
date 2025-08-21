import dataclasses
from collections.abc import Callable

import chex
import numpy as np
from jax import numpy as jnp
from jax import random as jr


def _forward_process(inputs, times, noise):
  new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
  times = times.reshape(new_shape)
  inputs_t = times * inputs + (1.0 - times) * noise
  return inputs_t


@dataclasses.dataclass
class FlowMatchingConfig:
  n_sampling_steps: int = 25
  time_eps: float = 1e-3
  time_max: float = 1.0


class FlowMatching:
  def __new__(cls, model_fn: Callable, config: FlowMatchingConfig):
    def loss_fn(params, rng_key, inputs, is_training, context=None, **kwargs):
      time_key, rng_key = jr.split(rng_key)
      times = jr.uniform(time_key, shape=(inputs.shape[0],))
      times = times * (config.time_max - config.time_eps) + config.time_eps
      noise_key, rng_key = jr.split(rng_key)
      noise = jr.normal(noise_key, inputs.shape)
      inputs_t = _forward_process(inputs, times, noise)
      vs = model_fn(
        variables={"params": params},
        rngs={"dropout": rng_key},
        inputs=inputs_t,
        times=times * 999.0,
        context=context,
        is_training=is_training,
      )
      target = inputs - noise
      loss = jnp.mean(jnp.square(target - vs))
      return loss

    def sample_fn(rng_key, state, sample_shape=(), context=None, **kwargs):
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
