import ml_collections


def new_dict(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  config = ml_collections.ConfigDict()
  config.rng_key = 1
  config.model = "flow_matching"
  config.score_net = "unet"
  n_classes = 10
  n_out_channels = 3
  batch_size = 512

  config.nn = new_dict(
    dit_score_net=new_dict(
      n_channels=768,
      n_out_channels=n_out_channels,
      patch_size=2,
      n_blocks=12,
      n_heads=12,
      dropout_rate=0.1,
      n_classes=n_classes,
    ),
    unet_score_net=new_dict(
      n_channels=128,
      n_out_channels=3,
      channel_multipliers=(1, 2, 2, 2),
      attention_resolutions=(16,),
      n_resnet_blocks=4,
      n_embedding=256,
      dropout_rate=0.1,
      n_classes=n_classes,
    ),
  )
  config.data = new_dict(
    dataset="cifar10",
    image_size=32,
    n_classes=n_classes,
    n_out_channels=n_out_channels,
  )
  config.training = new_dict(
    n_steps=500_000,
    batch_size=batch_size,
    buffer_size=batch_size * 10,
    prefetch_size=batch_size * 2,
    do_reshuffle=True,
    checkpoints=new_dict(
      max_to_keep=10,
      save_interval_steps=1,
    ),
    ema_rate=0.999,
    n_eval_frequency=10_000,
    n_checkpoint_frequency=30_000,
    n_eval_batches=10,
    n_sampling_frequency=50_000,
  )

  config.optimizer = new_dict(
    name="adam",
    params=new_dict(
      learning_rate=0.0002,
      weight_decay=1e-6,
      do_warmup=True,
      warmup_steps=1_000,
      do_decay=True,
      decay_steps=500_000,
      end_learning_rate=1e-6,
      init_learning_rate=1e-8,
      do_gradient_clipping=True,
      gradient_clipping=1.0,
    ),
  )

  return config
