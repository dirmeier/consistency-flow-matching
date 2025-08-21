import dataclasses
from collections.abc import Callable
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import erf


def _pseudo_huber_loss(inputs, targets):
  c = 0.00054 * jnp.sqrt(np.prod(inputs.shape[1:3]))
  return jnp.sqrt(jnp.square(inputs - targets) + c**2) - c


def _discretization_schedule(
  n_curr_train_step, n_total_train_steps, s0: int = 10, s1: int = 1280
):
  k_prime = n_total_train_steps / (jnp.log2(jnp.floor(s1 / s0)) + 1)
  k_prime = jnp.floor(k_prime)
  n_timesteps = s0 * jnp.power(2, jnp.floor(n_curr_train_step / k_prime))
  n_timesteps = jnp.minimum(n_timesteps, s1) + 1
  return jnp.ceil(n_timesteps).astype(jnp.int32)


def _sigmas(
  num_timesteps: int,
  sigma_min: float = 0.002,
  sigma_max: float = 80.0,
  rho: float = 7.0,
):
  """Standard EDM schedule/"""
  rho_inv = np.reciprocal(rho)
  steps = np.arange(num_timesteps) / (num_timesteps - 1)
  sigmas = sigma_min**rho_inv + steps * (
    sigma_max**rho_inv - sigma_min**rho_inv
  )
  sigmas = sigmas**rho
  return sigmas


def _loss_weighting(sigmas):
  return jnp.reciprocal(sigmas[1:] - sigmas[:-1])


def _timestep_schedule(rng_key, n_samples, sigmas, mean, std):
  logits = erf((jnp.log(sigmas[1:]) - mean) / (std * jnp.sqrt(2))) - erf(
    (jnp.log(sigmas[:-1]) - mean) / (std * jnp.sqrt(2))
  )
  timesteps = jr.categorical(rng_key, logits=logits, shape=(n_samples,))
  return timesteps


def _skip_scaling(sigma, sigma_data, sigma_min):
  return sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)


def _out_scaling(sigma, sigma_data, sigma_min):
  return (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5


def _forward(rng_key, params, apply_fn, inputs, sigma, sigma_data, sigma_min):
  new_shape = (-1,) + tuple(np.ones(inputs.ndim - 1, dtype=np.int32).tolist())
  preds = apply_fn(
    {"params": params},
    rngs={"dropout": rng_key},
    inputs=inputs,
    times=sigma,
  )
  c_skip = _skip_scaling(sigma, sigma_data, sigma_min)
  c_out = _out_scaling(sigma, sigma_data, sigma_min)
  return c_skip.reshape(new_shape) * inputs + c_out.reshape(new_shape) * preds


@dataclasses.dataclass
class ConsistencyMatchingConfig:
  n_total_train_steps: int
  sigma_min: float = 0.002
  sigma_max: float = 80.0
  rho: float = 7.0
  sigma_data: float = 0.5
  s0: int = 10
  s1: int = 1280
  lognormal_mean: float = -1.1
  lognormal_std: float = 2.0


class ConsistencyMatching:
  def __new__(cls, model_fn: Callable, config: ConsistencyMatchingConfig):
    def loss_fn(
      params,
      rng_key,
      inputs,
      sigmas,
      context=None,
      is_training=True,
    ):
      B, *_ = inputs.shape
      new_shape = (-1,) + tuple(
        np.ones(inputs.ndim - 1, dtype=np.int32).tolist()
      )
      score_model = partial(model_fn, context=context, is_training=is_training)

      timesteps_key, rng_key = jr.split(rng_key)
      timesteps = _timestep_schedule(
        timesteps_key, B, sigmas, config.lognormal_mean, config.lognormal_std
      )

      noise_key, rng_key = jr.split(rng_key)
      noise = jr.normal(noise_key, inputs.shape)
      current_sigmas, next_sigmas = sigmas[timesteps], sigmas[timesteps + 1]

      apply_key, rng_key = jr.split(rng_key)
      next_noisy_inputs = inputs + next_sigmas.reshape(new_shape) * noise
      pred_next = _forward(
        apply_key,
        params,
        score_model,
        inputs=next_noisy_inputs,
        sigma=next_sigmas,
        sigma_data=config.sigma_data,
        sigma_min=config.sigma_min,
      )

      apply_key, rng_key = jr.split(rng_key)
      current_noisy_inputs = inputs + current_sigmas.reshape(new_shape) * noise
      pred_current = _forward(
        apply_key,
        params,
        model_fn,
        inputs=current_noisy_inputs,
        sigma=current_sigmas,
        sigma_data=config.sigma_data,
        sigma_min=config.sigma_min,
      )
      pred_current = jax.lax.stop_gradient(pred_current)

      loss_weights = _loss_weighting(sigmas)[timesteps]
      loss = _pseudo_huber_loss(pred_next, pred_current)
      weighted_loss = loss * loss_weights.reshape(new_shape)
      return weighted_loss.mean()

    def sigma_schedule_fn(n_step):
      nk = _discretization_schedule(
        n_step,
        config.n_total_train_steps,
        config.s0,
        config.s1,
      )
      sigs = _sigmas(nk, config.sigma_min, config.sigma_max, config.rho)
      return sigs

    def sample_fn(
      rng_key,
      state,
      sample_shape=(),
      context=None,
      **kwargs,
    ):
      """One-step sampler"""
      yt = jr.normal(rng_key, sample_shape) * config.sigma_max
      sigmas = jnp.full(sample_shape[0], config.sigma_max)
      y0 = state.apply_fn(
        variables={"params": state.ema_params},
        inputs=yt,
        times=sigmas,
        context=context,
        is_training=False,
      )
      return y0

    return loss_fn, sigma_schedule_fn, sample_fn
