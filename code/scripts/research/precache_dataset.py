#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import time

from hydra import compose, initialize_config_dir
import omegaconf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import dataloader  # noqa: E402


omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', lambda: 1)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _dataset_len(dataset):
  try:
    return len(dataset)
  except TypeError:
    return 'streaming'


def _build_config(overrides):
  config_dir = PROJECT_ROOT / 'configs'
  with initialize_config_dir(
    version_base=None,
    config_dir=str(config_dir)):
    return compose(config_name='config', overrides=overrides)


def main():
  parser = argparse.ArgumentParser(
    description='Build and cache tokenized datasets without launching training.')
  parser.add_argument(
    '--skip-train',
    action='store_true',
    help='Only build the validation split.')
  parser.add_argument(
    '--skip-valid',
    action='store_true',
    help='Only build the training split.')
  parser.add_argument(
    'overrides',
    nargs='*',
    help='Hydra-style overrides, e.g. data=openwebtext-split model.length=1024')
  args = parser.parse_args()

  print('building config...', flush=True)
  start_time = time.time()
  config = _build_config(args.overrides)
  print(f'config built in {time.time() - start_time:.2f}s', flush=True)
  print(f'cache_dir: {config.data.cache_dir}', flush=True)
  print(f'train dataset: {config.data.train}', flush=True)
  print(f'valid dataset: {config.data.valid}', flush=True)
  print(
    'subset sizes: '
    f"train={config.data.get('train_max_examples', None)}, "
    f"valid={config.data.get('valid_max_examples', None)}",
    flush=True)
  tokenizer = dataloader.get_tokenizer(config)
  preprocessing_num_proc = config.data.get(
    'preprocessing_num_proc', config.loader.num_workers)
  print(f'preprocessing_num_proc: {preprocessing_num_proc}', flush=True)

  if not args.skip_train:
    train_dataset = dataloader.get_dataset(
      config.data.train,
      tokenizer,
      mode='train',
      wrap=config.data.wrap,
      insert_eos=config.data.insert_train_eos,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=preprocessing_num_proc,
      revision=config.data.get('train_revision', None),
      split_override=config.data.get('train_split', None),
      max_examples=config.data.get('train_max_examples', None))
    print(f'train examples: {_dataset_len(train_dataset)}', flush=True)

  if not args.skip_valid:
    if config.data.valid in ['text8', 'lm1b', 'ag_news']:
      validation_split = 'test'
    else:
      validation_split = 'validation'
    valid_dataset = dataloader.get_dataset(
      config.data.valid,
      tokenizer,
      mode=validation_split,
      wrap=config.data.wrap,
      insert_eos=config.data.insert_valid_eos,
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=preprocessing_num_proc,
      revision=config.data.get('valid_revision', None),
      split_override=config.data.get('valid_split', None),
      max_examples=config.data.get('valid_max_examples', None))
    print(f'valid examples: {_dataset_len(valid_dataset)}', flush=True)


if __name__ == '__main__':
  main()
