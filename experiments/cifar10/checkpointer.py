import os
import pickle
from typing import Any

import optax
import orbax.checkpoint as ocp
from absl import logging
from flax import core, struct
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax import tree_util


class EMATrainState(TrainState):
  ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)


class BatchStatsEMATrainState(TrainState):
  ema_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  batch_stats: Any


def new_train_state(variables, model, config):
  logging.info("get train state and checkpointer")
  if "batch_stats" not in variables:
    return EMATrainState.create(
      apply_fn=model.apply,
      params=variables["params"],
      ema_params=variables["params"].copy(),
      tx=get_optimizer(config),
    )
  else:
    ts = BatchStatsEMATrainState.create(
      apply_fn=model.apply,
      params=variables["params"],
      ema_params=variables["params"].copy(),
      batch_stats=variables["batch_stats"],
      tx=get_optimizer(config),
    )
  return ts


def get_optimizer(config):
  if config.params.do_warmup and config.params.do_decay:
    lr = optax.warmup_cosine_decay_schedule(
      init_value=config.params.init_learning_rate,
      peak_value=config.params.learning_rate,
      warmup_steps=config.params.warmup_steps,
      decay_steps=config.params.decay_steps,
      end_value=config.params.end_learning_rate,
    )
  elif config.params.do_warmup:
    lr = optax.linear_schedule(
      init_value=config.params.init_learning_rate,
      end_value=config.params.learning_rate,
      transition_steps=config.params.warmup_steps,
    )
  elif config.params.do_decay:
    lr = optax.cosine_decay_schedule(
      init_value=config.params.learning_rate,
      decay_steps=config.params.decay_steps,
      alpha=config.params.end_learning_rate / config.params.learning_rate,
    )
  else:
    lr = config.params.learning_rate

  if config.name == "adamw":
    tx = optax.adamw(lr, weight_decay=config.params.weight_decay)
  elif config.name == "radam":
    tx = optax.radam(lr)
  else:
    tx = optax.adam(lr)

  if config.params.do_gradient_clipping:
    tx = optax.chain(
      optax.clip_by_global_norm(config.params.gradient_clipping), tx
    )
  return tx


def get_checkpointer_fns(
  outfolder, config, model_config, criterion="train/loss"
):
  options = ocp.CheckpointManagerOptions(
    max_to_keep=config.max_to_keep,
    save_interval_steps=config.save_interval_steps,
    create=True,
  )
  checkpointer = ocp.PyTreeCheckpointer()
  checkpoint_manager = ocp.CheckpointManager(
    outfolder,
    checkpointer,
    options,
  )
  save_pickle(os.path.join(outfolder, "config.pkl"), model_config)

  def save_fn(step, ckpt, metrics):
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpoint_manager.save(
      step=step,
      items=ckpt,
      save_kwargs={"save_args": save_args},
      metrics=metrics,
    )
    checkpoint_manager.wait_until_finished()

  def restore_best_fn():
    return checkpoint_manager.restore(checkpoint_manager.best_step())

  def restore_last_fn():
    return checkpoint_manager.restore(checkpoint_manager.latest_step())

  return checkpoint_manager, save_fn, restore_best_fn, restore_last_fn


def get_latest_train_state(mngr, state):
  try:
    logging.info("trying to load train state")
    state = _restore_train_state(mngr, state, "latest")
    logging.info("successfully restored train state")
    return state, mngr.latest_step()
  except Exception as e:
    logging.info(str(e))
    logging.info("training from scratch")
    pass
  return state, 0


def _restore_train_state(mngr, ts, which):
  step = mngr.latest_step() if which == "latest" else mngr.best_step()
  restored_dict = mngr.restore(step)
  restored_optimizer = _restore_optimizer_state(
    ts.opt_state, restored_dict["opt_state"]
  )
  if "batch_stats" not in restored_dict:
    ts = ts.replace(
      params=restored_dict["params"],
      ema_params=restored_dict["ema_params"],
      step=restored_dict["step"],
      opt_state=restored_optimizer,
    )
  else:
    ts = ts.replace(
      params=restored_dict["params"],
      ema_params=restored_dict["ema_params"],
      batch_stats=restored_dict["batch_stats"],
      step=restored_dict["step"],
      opt_state=restored_optimizer,
    )
  return ts


def _restore_optimizer_state(opt_state, restored):
  return tree_util.tree_unflatten(
    tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
  )


def save_pickle(outfile, obj):
  with open(outfile, "wb") as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(outfile):
  with open(outfile, "rb") as handle:
    x = pickle.load(handle)
  return x
