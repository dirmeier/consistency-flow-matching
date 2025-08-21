from collections.abc import Callable

import numpy as np
from jax import numpy as jnp
from jax import random as jr


class DenoisingDiffusionConfig:
  sigma_min: float = 0.002
  sigma_max: float = 80.0
  rho: float = 7.0
  sigma_data: float = 0.5
  P_mean: float = -1.2
  P_std: float = 1.2
  S_churn: float = 40
  S_min: float = 0.05
  S_max: float = 50
  S_noise: float = 1.003
  n_sampling_steps: int = 25

  def sigma(self, eps):
    return jnp.exp(eps * self.P_std + self.P_mean)

  def loss_weight(self, sigma):
    return (jnp.square(sigma) + jnp.square(self.sigma_data)) / jnp.square(
      sigma * self.sigma_data
    )

  def skip_scaling(self, sigma):
    return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

  def out_scaling(self, sigma):
    return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

  def in_scaling(self, sigma):
    return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

  def noise_conditioning(self, sigma):
    return 0.25 * jnp.log(sigma + 1e-30)

  def sampling_sigmas(self, num_steps):
    rho_inv = 1 / self.rho
    step_idxs = jnp.arange(num_steps, dtype=jnp.float32)
    sigmas = (
      self.sigma_max**rho_inv
      + step_idxs
      / (num_steps - 1)
      * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
    ) ** self.rho
    return jnp.concatenate([sigmas, jnp.zeros_like(sigmas[:1])])

  def sigma_hat(self, sigma, num_steps):
    gamma = (
      jnp.minimum(self.S_churn / num_steps, 2**0.5 - 1)
      if self.S_min <= sigma <= self.S_max
      else 0
    )
    return sigma + gamma * sigma


class DenoisingDiffusion:
  def __new__(cls, model_fn: Callable, config: DenoisingDiffusionConfig()):
    def _denoise(rng_key, params, sample, sigma, context, is_training):
      # taken from https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/generate.py#L69
      new_shape = (-1,) + tuple(
        np.ones(sample.ndim - 1, dtype=np.int32).tolist()
      )
      # apply scaling
      inputs = sample * config.in_scaling(sigma).reshape(new_shape)
      sigma = config.noise_conditioning(sigma)
      out = model_fn(
        variables={"params": params},
        rngs={"dropout": rng_key},
        inputs=inputs,
        times=sigma,
        context=context,
        is_training=is_training,
      )
      # output scaling and skip connection
      skip = sample * config.skip_scaling(sigma).reshape(new_shape)
      outputs = out * config.out_scaling(sigma).reshape(new_shape)
      outputs = skip + outputs
      return outputs

    def loss_fn(params, rng_key, inputs, is_training, context=None, **kwargs):
      new_shape = (-1,) + tuple(
        np.ones(inputs.ndim - 1, dtype=np.int32).tolist()
      )

      epsilon_key, rng_key = jr.split(rng_key)
      epsilon = jr.normal(epsilon_key, (inputs.shape[0],))
      sigma = config.sigma(epsilon)

      noise_key, rng_key = jr.split(rng_key)
      noise = jr.normal(noise_key, inputs.shape)
      noise = noise * sigma.reshape(new_shape)

      denoise_key, rng_key = jr.split(rng_key)
      target_hat = _denoise(
        denoise_key,
        params,
        sample=inputs + noise,
        sigma=sigma,
        context=context,
        is_training=is_training,
      )
      loss = jnp.square(inputs - target_hat)
      loss = config.loss_weight(sigma).reshape(new_shape) * loss
      return jnp.mean(loss)

    def sample_fn(rng_key, state, sample_shape=(), context=None, **kwargs):
      """Heun sampler from EDM paper as a comparison."""
      n = sample_shape[0]
      noise_key, rng_key = jr.split(rng_key)
      sigmas = config.sampling_sigmas(config.n_sampling_steps)
      noise = jr.normal(noise_key, sample_shape) * sigmas[0]

      sample_next = noise
      for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        pred_key1, pred_key2, rng_key = jr.split(rng_key, 3)
        sample_curr = sample_next
        pred_curr = _denoise(
          pred_key1,
          state.ema_params,
          sample=sample_curr,
          sigma=jnp.repeat(sigma, n),
          context=context,
          is_training=False,
        )
        d_cur = (sample_curr - pred_curr) / sigma
        sample_next = sample_curr + d_cur * (sigma_next - sigma)

        # second order correction
        if i < config.n_sampling_steps - 1:
          pred_next = _denoise(
            pred_key2,
            state.ema_params,
            sample=sample_next,
            sigma=jnp.repeat(sigma_next, n),
            context=context,
            is_training=False,
          )
          d_prime = (sample_next - pred_next) / sigma_next
          sample_next = sample_curr + (sigma_next - sigma) * (
            0.5 * d_cur + 0.5 * d_prime
          )
      return sample_next

    return loss_fn, lambda x: x, sample_fn
