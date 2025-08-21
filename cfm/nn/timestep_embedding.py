from jax import numpy as jnp


def timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
  half = embedding_dim // 2
  freqs = jnp.exp(-jnp.log(10_000) * jnp.arange(0, half) / half)
  emb = timesteps.astype(dtype)[:, None] * freqs[None, ...]
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
  return emb
