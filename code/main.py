import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils

from datetime import datetime

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

if hasattr(torch, 'set_float32_matmul_precision'):
  torch.set_float32_matmul_precision('high')


def _normalize_runtime_config(config):
  optim_name = str(
    getattr(config.optim, 'name', 'adamw')).lower()
  if optim_name != 'sam':
    return

  clip_value = getattr(config.trainer, 'gradient_clip_val', 0.0)
  manual_clip_value = getattr(
    config.training, 'manual_gradient_clip_val', None)
  if manual_clip_value is None:
    config.training.manual_gradient_clip_val = clip_value
  manual_accumulate_grad_batches = getattr(
    config.training, 'manual_accumulate_grad_batches', None)
  if manual_accumulate_grad_batches is None:
    config.training.manual_accumulate_grad_batches = (
      config.trainer.accumulate_grad_batches)
  config.trainer.gradient_clip_val = 0.0
  config.trainer.accumulate_grad_batches = 1


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)

def _compute_sequence_entropy(token_ids, vocab_size):
  import numpy as np
  if isinstance(token_ids, torch.Tensor):
    token_ids = token_ids.cpu().numpy()
  unique, counts = np.unique(token_ids, return_counts=True)
  probs = counts / len(token_ids)
  # entropy = -np.sum(probs * np.log2(probs + 1e-10))
  entropy = -np.sum(probs * np.log(probs + 1e-10))
  
  return entropy

def _generate_samples(diffusion_model, config, logger,
                      tokenizer, prompt=None):
  logger.info('Starting Sample Eval.')
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_samples = []
  all_nfe = []
  all_entropy = []
  total_sampling_time = 0.0  # Track total sampling time
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      # Time the sampling
      start_time = torch.cuda.Event(enable_timing=True)
      end_time = torch.cuda.Event(enable_timing=True)
      start_time.record()

      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)

      end_time.record()
      # Wait for GPU to finish and calculate elapsed time
      torch.cuda.synchronize()
      batch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
      total_sampling_time += batch_time

      model.metrics.record_entropy(samples)
      text_samples = model.tokenizer.batch_decode(samples)
      model.metrics.record_generative_perplexity(
        text_samples, config.model.length, device=model.device)
      all_samples.extend(list(text_samples))
      all_entropy.extend([
        _compute_sequence_entropy(sample, model.tokenizer.vocab_size)
        for sample in samples
      ])
      if not config.sampling.early_stopping:
        all_nfe.extend([config.sampling.steps] * len(text_samples))
      else:
        all_nfe.extend(model._last_completion_steps.cpu().tolist())
  generative_ppl = 0.
  entropy = 0.
  if not config.sampling.semi_ar:
    generative_ppl = model.metrics.gen_ppl.compute().item()
    entropy = model.metrics.sample_entropy.compute().item()
    print('Generative perplexity:', generative_ppl)
    print('Sample entropy:', entropy)

  # samples_path = config.eval.generated_samples_path
  # with fsspec.open(samples_path, 'w') as f:
  #   json.dump({'generative_ppl': generative_ppl,
  #              'entropy': entropy,
  #              'generated_seqs': all_samples}, f, indent=4)
  # print('Samples saved at:', samples_path)

  # Save to JSON if path is provided
  save_path = getattr(config.eval, 'generated_samples_path', None)
  if save_path:
    avg_time_per_sample = (
      total_sampling_time / len(all_samples)
      if all_samples else 0.0)
    samples_per_second = (
      len(all_samples) / total_sampling_time
      if total_sampling_time > 0 else 0.0)
    avg_nfe = sum(all_nfe) / len(all_nfe) if all_nfe else 0.0
    results = {
      'metadata': {
        'timestamp': datetime.now().isoformat(),
        'model_path': config.eval.checkpoint_path,
        'sampler': config.sampling.predictor,
        'sample_steps': config.sampling.steps,
        'num_sample_batches': config.sampling.num_sample_batches,
        'batch_size': config.loader.eval_batch_size,
        'total_samples': len(all_samples),
        'prompt': prompt,
        'semi_ar': config.sampling.semi_ar,
        'stride_length': stride_length if config.sampling.semi_ar else None,
        'num_strides': num_strides if config.sampling.semi_ar else None,
        'use_float64': config.sampling.use_float64,
        'gumbel_precision': config.sampling.gumbel_precision,
      },
      'performance': {
        'generative_perplexity': generative_ppl,
        'total_generation_time_seconds': total_sampling_time,
        'average_time_per_sample': avg_time_per_sample,
        'samples_per_second': samples_per_second,
        'avg_nfe': avg_nfe,
        'avg_entropy': entropy,
        'avg_entropy_per_sequence': sum(all_entropy) / len(all_entropy) if all_entropy else 0.0,
      },
      'samples': all_samples,
      'nfe': all_nfe,
    }
    save_dir = os.path.dirname(save_path)
    if save_dir:
      os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'w') as f:
      json.dump(results, f, indent=2)
    print(f"\nResults saved to: {save_path}")

def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config)
  else:
    model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  _normalize_runtime_config(config)
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'cdlm':
    diffusion_model = algo.CDLM
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'd3pm':
    diffusion_model = algo.D3PMAbsorb
  elif config.algo.name == 'sedd':
    diffusion_model = algo.SEDDAbsorb
  elif config.algo.name == 'duo':
    diffusion_model = algo.DUO
  elif config.algo.name == 'distillation':
    diffusion_model = algo.Distillation
  elif config.algo.name == 'ot-finetune':
    diffusion_model = algo.OptimalTransportFinetune
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  else:
    _train(**kwargs)


if __name__ == '__main__':
  main()
