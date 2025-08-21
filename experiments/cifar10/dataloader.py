import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from jax import numpy as jnp
from jax import random as jr


def data_loaders(rng_key, config, split="train", outpath: str = None):
  datasets = tfds.load(
    config.data.dataset,
    try_gcs=False,
    split=split,
    data_dir=outpath,
  )
  assert config.training.batch_size % jax.device_count() == 0
  if isinstance(split, str):
    datasets = [datasets]
  itrs = []
  for dataset in datasets:
    itr_key, rng_key = jr.split(rng_key)
    itr = _as_batched_numpy_iter(itr_key, dataset, config)
    itrs.append(itr)
  return itrs


def _crop_resize(image, resolution):
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  crop = tf.minimum(h, w)
  image = image[
    (h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2
  ]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC,
  )
  return tf.cast(image, tf.float32)


def _as_batched_numpy_iter(rng_key, itr, config):
  def _process_flowers(batch):
    img = tf.cast(batch["image"], tf.float32) / 255.0
    img = _crop_resize(img, config.data.image_size)
    img = 2.0 * img - 1.0
    return {"image": img}

  def _process_cifar(batch):
    img = tf.cast(batch["image"], tf.float32) / 255.0
    img = 2.0 * img - 1.0
    return {"image": img, "label": batch["label"]}

  def _process__mnist(batch):
    img = tf.cast(batch["image"], tf.float32) / 255.0
    img = _crop_resize(img, config.data.image_size)
    img = 2.0 * img - 1.0
    return {"image": img, "label": batch["label"]}

  logging.info(f"using {config.data.dataset}")
  if config.data.dataset == "cifar10":
    process_fn = _process_cifar
  elif config.data.dataset == "mnist":
    process_fn = _process__mnist
  elif config.data.dataset == "oxford_flowers102":
    process_fn = _process_flowers
  else:
    raise ValueError("data set not found")

  max_int32 = jnp.iinfo(jnp.int32).max
  seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)
  return (
    itr.repeat()
    .shuffle(
      config.training.buffer_size,
      reshuffle_each_iteration=config.training.do_reshuffle,
      seed=int(seed),
    )
    .map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(config.training.batch_size, drop_remainder=True)
    .prefetch(config.training.prefetch_size)
    .as_numpy_iterator()
  )
