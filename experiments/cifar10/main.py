import hashlib
import os

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from absl import app, flags, logging
from checkpointer import (
  get_checkpointer_fns,
  get_latest_train_state,
  new_train_state,
)
from dataloader import data_loaders
from flax import jax_utils
from flax.training import common_utils
from jax import random as jr
from jax.lib import xla_bridge
from ml_collections import config_flags

from cfm import (
  ConsistencyFlowMatching,
  ConsistencyFlowMatchingConfig,
  ConsistencyMatching,
  ConsistencyMatchingConfig,
  DenoisingDiffusion,
  DenoisingDiffusionConfig,
  FlowMatching,
  FlowMatchingConfig,
)
from cfm.nn import DiT, UNet

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "model configuration")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_bool("usewand", False, "use wandb for logging")
flags.mark_flags_as_required(["workdir", "config"])


def get_model_and_loss(config):
  if config.score_net == "dit":
    logging.info("using dit")
    score_model = DiT(**config.nn.dit_score_net.to_dict())
  else:
    logging.info("using unet")
    score_model = UNet(**config.nn.unet_score_net.to_dict())
  if config.model == "consistency_matching":
    matching_fns = ConsistencyMatching(
      score_model.apply,
      ConsistencyMatchingConfig(config.training.n_steps),
    )
  elif config.model == "flow_matching":
    matching_fns = FlowMatching(score_model.apply, FlowMatchingConfig())
  elif config.model == "consistency_flow_matching":
    matching_fns = ConsistencyFlowMatching(
      score_model.apply, ConsistencyFlowMatchingConfig()
    )
  elif config.model == "denoising_diffusion":
    matching_fns = DenoisingDiffusion(
      score_model.apply, DenoisingDiffusionConfig()
    )
  else:
    raise ValueError("dont find model")
  return score_model, matching_fns


def get_step_fn(loss_fn):
  @jax.jit
  def step_fn(rngs, state, batch, sigmas):
    loss, grads = jax.value_and_grad(loss_fn)(
      state.params,
      rngs,
      inputs=batch["image"],
      context=batch["label"],
      sigmas=sigmas,
      is_training=True,
    )

    loss = jax.lax.pmean(loss, axis_name="batch")
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    new_ema_params = optax.incremental_update(
      new_state.params,
      new_state.ema_params,
      step_size=1.0 - FLAGS.config.training.ema_rate,
    )
    new_state = new_state.replace(ema_params=new_ema_params)
    return {"loss": loss}, new_state

  return step_fn


def get_eval_fn(loss_fn):
  @jax.jit
  def eval_fn(rngs, state, batch, sigmas):
    loss = loss_fn(
      state.params,
      rngs,
      inputs=batch["image"],
      context=batch["label"],
      sigmas=sigmas,
      is_training=False,
    )
    loss = jax.lax.pmean(loss, axis_name="batch")
    return {"loss": loss}

  return eval_fn


def init_model(rng_key, model, train_iter):
  batch = next(iter(train_iter))
  batch = jax.tree.map(lambda x: x[:10], batch)
  times = jnp.zeros(10)
  params = model.init(
    {"params": rng_key},
    inputs=batch["image"],
    times=times,
    context=batch["label"],
    is_training=False,
  )
  return params


def metrics_to_summary(train_metrics, val_metrics):
  train_metrics = common_utils.get_metrics(train_metrics)
  val_metrics = common_utils.get_metrics(val_metrics)
  train_summary = {
    f"train/{k}": v
    for k, v in jax.tree.map(lambda x: float(x.mean()), train_metrics).items()
  }
  val_summary = {
    f"val/{k}": v
    for k, v in jax.tree.map(lambda x: float(x.mean()), val_metrics).items()
  }
  return train_summary | val_summary


def train(rng_key, model, matching_fns, config, train_iter, val_iter, model_id):
  init_key, rng_key = jr.split(rng_key)
  state = new_train_state(
    init_model(init_key, model, train_iter), model, config.optimizer
  )
  mngr, ckpt_save_fn, *_ = get_checkpointer_fns(
    os.path.join(FLAGS.workdir, "checkpoints", model_id),
    config.training.checkpoints,
    config.nn.to_dict(),
  )
  state, cstep = get_latest_train_state(mngr, state)

  loss_fn, sigma_fn, sampling_fn = matching_fns
  pstep_fn = jax.pmap(
    get_step_fn(loss_fn),
    axis_name="batch",
  )
  peval_fn = jax.pmap(
    get_eval_fn(loss_fn),
    axis_name="batch",
  )
  pstate = jax_utils.replicate(state)

  logging.info("training model")
  step_key, rng_key = jr.split(rng_key)
  train_metrics = []
  logging.info(f"starting/resuming training at step: {cstep}")
  for step, batch in zip(
    range(cstep + 1, config.training.n_steps + 1), train_iter
  ):
    train_key, val_key, sample_key = jr.split(jr.fold_in(step_key, step), 3)
    psigmas = jax_utils.replicate(sigma_fn(step))
    pbatch = common_utils.shard(batch)
    if step == 1 and jax.process_index() == 0:
      logging.info(f"pbatch shape: {pbatch['image'].shape}")
    metrics, pstate = pstep_fn(
      jr.split(train_key, jax.device_count()), pstate, pbatch, psigmas
    )
    train_metrics.append(metrics)
    is_first_or_last_step = step == config.training.n_steps or step == 1
    if step % config.training.n_eval_frequency == 0 or is_first_or_last_step:
      val_metrics = []
      for val_idx, batch in zip(
        range(config.training.n_eval_batches), val_iter
      ):
        pbatch = common_utils.shard(batch)
        metrics = peval_fn(
          jr.split(jr.fold_in(val_key, val_idx), jax.device_count()),
          pstate,
          pbatch,
          psigmas,
        )
        val_metrics.append(metrics)
      summary = metrics_to_summary(train_metrics, val_metrics)
      train_metrics = []
      if jax.process_index() == 0:
        logging.info(
          f"loss at step {step}: {summary['train/loss']}/{summary['val/loss']}"
        )
      if (
        jax.process_index() == 0
        and step % config.training.n_checkpoint_frequency == 0
      ):
        ckpt_save_fn(
          step,
          jax_utils.unreplicate(pstate),
          summary,
        )
      if FLAGS.usewand and jax.process_index() == 0:
        wandb.log(summary, step=step)
    if (
      step % config.training.n_sampling_frequency == 0
      and jax.process_index() == 0
    ):
      log_images(
        sample_key, jax_utils.unreplicate(pstate), step, model_id, sampling_fn
      )
  if jax.process_index() == 0:
    sample_key, rng_key = jr.split(rng_key)
    log_images(
      sample_key,
      jax_utils.unreplicate(pstate),
      cstep + 1,
      model_id,
      sampling_fn,
    )
  logging.info("finished training")


def plot_figures(samples):
  img_size = FLAGS.config.data.image_size
  n_chan = FLAGS.config.data.n_out_channels
  n_samples = samples.shape[0]
  n_row, n_col = 12, 32
  chex.assert_equal(n_samples, n_row * n_col)

  def convert_batch_to_image_grid(image_batch):
    reshaped = (
      image_batch.reshape(n_row, n_col, img_size, img_size, n_chan)
      .transpose([0, 2, 1, 3, 4])
      .reshape(n_row * img_size, n_col * img_size, n_chan)
    )
    # undo intitial scaling, i.e., map [-1, 1] -> [0, 1]
    return reshaped / 2.0 + 0.5

  samples = convert_batch_to_image_grid(samples)
  fig = plt.figure(figsize=(16, 6))
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(
    samples,
    interpolation="nearest",
    cmap="gray",
  )
  plt.axis("off")
  plt.tight_layout()
  return fig


def log_images(rng_key, state, step, model_id, sampling_fn):
  logging.info("sampling images")
  img_size = FLAGS.config.data.image_size
  n_chan = FLAGS.config.data.n_out_channels

  batch_size = 64
  n_samples = batch_size * 6

  @jax.jit
  def sample(rng_key, state, context):
    samples = sampling_fn(
      rng_key,
      state,
      sample_shape=(batch_size, img_size, img_size, n_chan),
      context=context,
    )
    return samples

  all_samples = []
  for i in range(n_samples // batch_size):
    if "n_classes" in FLAGS.config.data:
      sample_key, rng_key = jr.split(rng_key)
      context = jr.choice(
        sample_key,
        FLAGS.config.data.n_classes,
        (batch_size,),
        replace=True,
      )
    else:
      context = None
    sample_key, rng_key = jr.split(rng_key)
    samples = sample(sample_key, state, context)
    all_samples.append(samples)
  all_samples = np.concatenate(all_samples, axis=0)
  fig = plot_figures(all_samples)

  if FLAGS.usewand:
    wandb.log({"images": wandb.Image(fig)}, step=step)

  for dpi in [200]:
    fl = os.path.join(
      FLAGS.workdir, "figures", f"{model_id}-sampled-{step}-dpi-{dpi}.png"
    )
    fig.savefig(fl, dpi=dpi)


def hash_value(config):
  h = hashlib.new("sha256")
  h.update(str(config).encode("utf-8"))
  return h.hexdigest()


def init_and_log_jax_env(tm):
  logging.set_verbosity(logging.INFO)
  logging.info("file prefix: %s", tm)
  logging.info("----- Checking JAX installation ----")
  logging.info(jax.devices())
  logging.info(jax.default_backend())
  logging.info(jax.device_count())
  logging.info(xla_bridge.get_backend().platform)
  logging.info("------------------------------------")
  return tm


def main(argv):
  del argv
  config = FLAGS.config.to_dict()
  model_id = f"{hash_value(config)}"
  init_and_log_jax_env(model_id)

  if FLAGS.usewand:
    wandb.init(
      project="cfm-experiment",
      config=config,
      dir=os.path.join(FLAGS.workdir, "wandb"),
    )
    wandb.run.name = model_id

  rng_key = jr.PRNGKey(FLAGS.config.rng_key)
  data_key, train_key, rng_key = jr.split(rng_key, 3)
  train_iter, val_iter = data_loaders(
    rng_key=data_key,
    config=FLAGS.config,
    split=["train[:95%]", "train[95%:]"],
    outpath=os.path.join(FLAGS.workdir, "data"),
  )

  model, matching_fns = get_model_and_loss(FLAGS.config)
  train(
    train_key,
    model,
    matching_fns,
    FLAGS.config,
    train_iter,
    val_iter,
    model_id,
  )


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
