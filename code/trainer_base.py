import itertools
import warnings
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers

import dataloader
import metrics
import models
import sam
import utils

from tqdm import trange

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  prior_loss: torch.FloatTensor
  num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

  def forward(self, t):
    t = (1 - self.eps) * t
    alpha_t = 1 - t 
    dalpha_t = - (1 - self.eps)
    return dalpha_t, alpha_t


# def sample_categorical(categorical_probs):
#   gumbel_norm = (
#     1e-10
#     - (torch.rand_like(categorical_probs) + 1e-10).log())
#   return (categorical_probs / gumbel_norm).argmax(dim=-1)

def _sampling_dtype(precision):
  precision = str(precision).lower()
  if precision in {'float64', 'fp64', 'double'}:
    return torch.float64
  if precision in {'float32', 'fp32', 'single'}:
    return torch.float32
  if precision in {'float16', 'fp16', 'half'}:
    return torch.float16
  raise ValueError(f'Unknown categorical sampling precision: {precision}')


def sample_categorical(categorical_probs, precision='float32'):
  precision_dtype = _sampling_dtype(precision)
  if precision_dtype == torch.float64:
    gumbel_norm = (
      1e-10
      - (torch.rand_like(categorical_probs, dtype=torch.float64) + 1e-10).log())
    return (categorical_probs.to(torch.float64) / gumbel_norm).argmax(dim=-1)
  if precision_dtype == torch.float32:
    gumbel_norm = (
      1e-10
      - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)
  # In half precision, 1e-10 underflows to zero. Clamp the fp16 uniform
  # variate at the smallest normal half value to avoid log(0) while keeping
  # the categorical kernel genuinely half precision.
  uniform = torch.rand_like(categorical_probs, dtype=torch.float16)
  uniform = uniform.clamp_min(torch.finfo(torch.float16).tiny)
  gumbel_norm = -uniform.log()
  return (
    categorical_probs.to(torch.float16) / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer,
    vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    if hasattr(self.config.algo, 'ignore_bos'):
      self.ignore_bos = config.algo.ignore_bos
    else:
      self.ignore_bos = False
    if hasattr(self.config.algo, 'loss_type'):
      self.loss_type = config.algo.loss_type
    self.tokenizer = tokenizer
    if vocab_size is None:
      self.vocab_size = len(self.tokenizer)
    else:
      self.vocab_size = vocab_size
    self.sampler = self.config.sampling.predictor
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.parameterization = self.config.algo.parameterization
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampling.p_nucleus
    # Noise Schedule
    self.noise = LogLinear()

    self.metrics = metrics.Metrics(
      gen_ppl_eval_model_name_or_path=\
        self.config.eval.gen_ppl_eval_model_name_or_path,
      eval_ppl_batch_size=\
        self.config.eval.perplexity_batch_size)

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.optim_name = getattr(
      self.config.optim, 'name', 'adamw').lower()
    self.use_sam = self.optim_name == 'sam'
    self.automatic_optimization = not self.use_sam
    self._sam_batch_buffer = []
    self._sam_warned_pending_validation = False
    manual_accumulate_grad_batches = getattr(
      self.config.training,
      'manual_accumulate_grad_batches',
      None)
    if manual_accumulate_grad_batches is None:
      manual_accumulate_grad_batches = 1
    self.manual_accumulate_grad_batches = int(
      manual_accumulate_grad_batches)
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _validate_configuration(self):
    assert self.config.algo.backbone in {'dit', 'hf_dit'}
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
      assert self.config.prior.type == 'none'

    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    return self

  def q_xt(self, x, alpha_t):
    raise NotImplementedError

  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(),
                           self.noise.parameters())

  def _effective_accumulate_grad_batches(self):
    if self.use_sam:
      return self.manual_accumulate_grad_batches
    return self.trainer.accumulate_grad_batches

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    self.noise.train()

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed']
    # is 1 iteration behind, so we're using the optimizer's progress.
    fit_loop = checkpoint['loops']['fit_loop']
    if 'epoch_loop.manual_optimization.optim_step_progress' in fit_loop:
      step_progress = fit_loop[
        'epoch_loop.manual_optimization.optim_step_progress']
      total_completed = step_progress['total']['completed']
      current_completed = step_progress['current']['completed']
    else:
      step_progress = fit_loop[
        'epoch_loop.automatic_optimization.optim_progress']
      total_completed = step_progress['optimizer']['step'][
        'total']['completed']
      current_completed = step_progress['optimizer']['step'][
        'current']['completed']
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = (
          total_completed * self._effective_accumulate_grad_batches())
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = (
          current_completed * self._effective_accumulate_grad_batches())
    # _batches_that_stepped tracks the number of global steps,
    # not the number of local steps, so we don't multiply with
    # self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = total_completed
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=self.config.loader.num_workers > 0))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    raise NotImplementedError

  def forward(self, xt, sigma):
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output = self.backbone(xt, sigma)
    return self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)

  def on_train_epoch_start(self):
    self.metrics.reset()
    self._sam_batch_buffer = []
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    if self.use_sam:
      self._sam_batch_buffer.append(batch)
      if not self._sam_should_step(batch_idx):
        return None
      return self._sam_update()

    current_accumulation_step = (
      batch_idx % self._effective_accumulate_grad_batches())
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'],
                        current_accumulation_step,
                        train_mode=True)
    self.metrics.update_train(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_train_epoch_end(self):
    for k, v in self.metrics.train_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)

  def on_validation_epoch_start(self):
    if self.use_sam and self._sam_batch_buffer:
      if not self._sam_warned_pending_validation:
        warnings.warn(
          'Validation started with pending SAM micro-batches. '
          'Carrying the buffered micro-batches across the validation '
          'boundary so the next SAM step still sees the full effective '
          'batch. For exact mid-epoch resume semantics, prefer a '
          'validation interval divisible by the effective SAM '
          'accumulation factor.',
          RuntimeWarning)
        self._sam_warned_pending_validation = True
    self.metrics.reset_valid()
    self._eval_mode()
    self._on_validation_epoch_start_extra()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    self._on_validation_step_extra(batch, batch_idx)
    return losses.loss

  def _get_sample_eval_steps(self):
    sample_eval_steps = getattr(self.config.eval,
                                'sample_eval_steps', [])
    if sample_eval_steps is None:
      return []
    return [int(step) for step in sample_eval_steps]

  def _sample_quality_for_step_count(self, sampling_steps):
    if self.config.sampling.num_sample_batches <= 0:
      return None, None, None

    entropy_metric = torchmetrics.aggregation.MeanMetric().to(
      self.device)
    gen_ppl_metric = None
    if self.config.eval.compute_generative_perplexity:
      gen_ppl_metric = metrics.Perplexity()
      if 'llama2' not in self.config.eval.gen_ppl_eval_model_name_or_path:
        gen_ppl_metric = gen_ppl_metric.to(self.device)

    last_text_samples = None
    for _ in range(self.config.sampling.num_sample_batches):
      samples = self.generate_samples(
        num_samples=self.config.loader.eval_batch_size,
        num_steps=sampling_steps)
      self.metrics.record_entropy(samples, metric=entropy_metric)
      last_text_samples = self.tokenizer.batch_decode(samples)
      if gen_ppl_metric is not None:
        self.metrics.record_generative_perplexity(
          last_text_samples, self.num_tokens, device=self.device,
          metric=gen_ppl_metric)

    return (last_text_samples,
            entropy_metric.compute(),
            gen_ppl_metric.compute() if gen_ppl_metric is not None
            else None)

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)
    self._on_validation_epoch_end_extra()
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      sample_eval_steps = self._get_sample_eval_steps()
      if sample_eval_steps:
        table_step = (
          self.config.sampling.steps
          if self.config.sampling.steps in sample_eval_steps
          else sample_eval_steps[0])
        for sampling_steps in sample_eval_steps:
          text_samples, entropy_value, gen_ppl_value = (
            self._sample_quality_for_step_count(
              sampling_steps=sampling_steps))
          if text_samples is None:
            continue

          if (self.trainer.global_rank == 0
              and hasattr(self.trainer.logger, 'log_table')
              and sampling_steps == table_step):
            self.trainer.logger.log_table(
              key=f'samples@global_step{self.global_step}',
              columns=['Generated Samples'],
              data=[[s] for s in text_samples[
                : self.config.sampling.num_sample_log]])

          self.log(f'val/sample_entropy_s{sampling_steps}',
                   entropy_value,
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
          if gen_ppl_value is not None:
            self.log(f'val/gen_ppl_s{sampling_steps}',
                     gen_ppl_value,
                     on_epoch=True,
                     on_step=False,
                     sync_dist=True)

          if (sampling_steps == self.config.sampling.steps
              or len(sample_eval_steps) == 1):
            self.log('val/sample_entropy',
                     entropy_value,
                     on_epoch=True,
                     on_step=False,
                     sync_dist=True)
            if gen_ppl_value is not None:
              self.log('val/gen_ppl',
                       gen_ppl_value,
                       on_epoch=True,
                       on_step=False,
                       sync_dist=True)
      else:
        samples, text_samples = None, None
        for _ in range(
          self.config.sampling.num_sample_batches):
          samples = self.generate_samples(
            num_samples=self.config.loader.eval_batch_size)
          
          self.metrics.record_entropy(samples)
          # Decode the samples to be re-tokenized by eval model
          text_samples = self.tokenizer.batch_decode(samples)
          if self.config.eval.compute_generative_perplexity:
            self.metrics.record_generative_perplexity(
              text_samples, self.num_tokens, device=self.device)
        if text_samples is not None:
          if self.trainer.global_rank == 0 and hasattr(
            self.trainer.logger, 'log_table'):
            # Log the last generated samples
            text_samples = text_samples[
              : self.config.sampling.num_sample_log]
            self.trainer.logger.log_table(
              key=f'samples@global_step{self.global_step}',
              columns=['Generated Samples'],
              data=[[s] for s in text_samples])
          if self.config.eval.compute_generative_perplexity:
            self.log('val/gen_ppl',
                     self.metrics.gen_ppl.compute(),
                     on_epoch=True,
                     on_step=False,
                     sync_dist=True)
            self.log('val/sample_entropy',
                     self.metrics.sample_entropy.compute(),
                     on_epoch=True,
                     on_step=False,
                     sync_dist=True)
    self._train_mode()

  def _on_validation_epoch_start_extra(self):
    return None

  def _on_validation_step_extra(self, batch, batch_idx):
    del batch, batch_idx
    return None

  def _on_validation_epoch_end_extra(self):
    return None

  def _sam_should_step(self, batch_idx):
    if len(self._sam_batch_buffer) >= self._effective_accumulate_grad_batches():
      return True
    num_training_batches = getattr(
      self.trainer, 'num_training_batches', None)
    return (
      isinstance(num_training_batches, int)
      and batch_idx + 1 >= num_training_batches)

  def _get_manual_optimizer(self):
    optimizer = self.optimizers()
    if isinstance(optimizer, (list, tuple)):
      optimizer = optimizer[0]
    return optimizer

  def _step_manual_schedulers(self):
    schedulers = self.lr_schedulers()
    if schedulers is None:
      return
    if not isinstance(schedulers, (list, tuple)):
      schedulers = [schedulers]
    for scheduler in schedulers:
      scheduler.step()

  def _clip_gradients_for_optimizer(self):
    clip_value = getattr(
      self.config.training,
      'manual_gradient_clip_val',
      None)
    if clip_value is None:
      clip_value = getattr(self.config.trainer,
                           'gradient_clip_val', 0.0)
    if clip_value is None or clip_value <= 0:
      return
    torch.nn.utils.clip_grad_norm_(
      list(self._get_parameters()), max_norm=clip_value)

  def _sam_update(self):
    buffered_batches = self._sam_batch_buffer
    self._sam_batch_buffer = []
    if not buffered_batches:
      return None

    optimizer = self._get_manual_optimizer()
    num_micro_batches = len(buffered_batches)
    optimizer.zero_grad()

    detached_losses = []
    for accum_step, batch in enumerate(buffered_batches):
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          accum_step,
                          train_mode=True)
      self.metrics.update_train(losses.nlls, losses.prior_loss,
                                losses.num_tokens)
      detached_losses.append(losses.loss.detach())
      self.manual_backward(losses.loss / num_micro_batches)

    self._clip_gradients_for_optimizer()

    def closure():
      for accum_step, batch in enumerate(buffered_batches):
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            accum_step,
                            train_mode=True)
        self.manual_backward(losses.loss / num_micro_batches)
      self._clip_gradients_for_optimizer()
      return torch.stack(detached_losses).mean()

    optimizer.step(closure=closure)
    optimizer.zero_grad()
    self._step_manual_schedulers()
    if self.ema:
      self.ema.update(self._get_parameters())

    avg_loss = torch.stack(detached_losses).mean()
    self.log(name='trainer/loss',
             value=avg_loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return avg_loss

  def configure_optimizers(self):
    optimizer_kwargs = dict(
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)
    if self.use_sam:
      optimizer = sam.SAM(
        self._get_parameters(),
        torch.optim.AdamW,
        rho=self.config.optim.sam_rho,
        adaptive=self.config.optim.sam_adaptive,
        **optimizer_kwargs)
      scheduler_optimizer = optimizer
    else:
      optimizer = torch.optim.AdamW(
        self._get_parameters(),
        **optimizer_kwargs)
      scheduler_optimizer = optimizer

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=scheduler_optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'name': 'trainer/lr'}
    if not self.use_sam:
      scheduler_dict['monitor'] = 'val/loss'
    return [optimizer], [scheduler_dict]

  def generate_samples(self, num_samples, num_steps, eps):
    raise NotImplementedError

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    self._eval_mode()
    samples = self.generate_samples(
      num_samples=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    self._train_mode()
    return samples

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step=None, train_mode=False):
    raise NotImplementedError

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False):
    (input_tokens, output_tokens,
     valid_tokens) = self._process_model_input(
       x0, valid_tokens)
    loss = self.nll(input_tokens, output_tokens,
                    current_accumulation_step, train_mode)
    assert loss.ndim == 2
    if self.ignore_bos:
      loss[:, 1:] = loss[:, 1:]
      valid_tokens[:, 1:] = valid_tokens[:, 1:]

    nlls = (loss * valid_tokens).sum()
    num_tokens = valid_tokens.sum()
    token_nll = nlls / num_tokens

    return Loss(loss=token_nll,
                nlls=nlls,
                prior_loss=0.0,
                num_tokens=num_tokens)


class Diffusion(TrainerBase):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.noise_removal in {
      'none', 'ancestral', 'greedy'}
    assert self.loss_type in {'elbo', 'low_var'}
    if self.config.sampling.noise_removal == 'greedy':
      assert self.sampler != 'analytic'
      assert self.parameterization in {'mean', 'subs'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def _process_sigma(self, sigma):
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    if self.T > 0:
      # For finite-T training, sample the deployment grid exactly instead of
      # drawing continuous times and quantizing them afterwards. This is a
      # closer match to the intended discrete-budget objective.
      t = (torch.floor(_eps_t * self.T) + 1).to(_eps_t.dtype) / self.T
    else:
      t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self._effective_accumulate_grad_batches())[
        accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype,
                     device=self.device)
    sigma_t0 = self._sigma_from_alphat(self.noise(t0)[1])
    model_output_t0 = self.forward(x0, sigma_t0)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)


  def nll_per_token(self, model_output, xt, x0, alpha_t,
                    dalpha_t, low_var):
    raise NotImplementedError

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t)
    log_x_theta = self.forward(xt, sigma=sigma)
    utils.print_nans(log_x_theta, 'model_output')
    return self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var')

  def _get_score(self, **kwargs):
    del kwargs
    raise NotImplementedError

  def _denoiser_update(self, x, t):
    raise NotImplementedError

  def _analytic_update(self, x, t, dt):
    raise NotImplementedError

  def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
    raise NotImplementedError

  ###
  def _get_num_transfer_tokens(self, mask_index, steps):
    raise NotImplementedError

  def _llada_update(self, x, t, mask_region, num_transfer_tokens, temperature, remasking_strategy, threshold):
    raise NotImplementedError

  def _sample_llada(self, num_samples, num_steps=None, eps=1e-5):
    raise NotImplementedError
  ###

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None,
                       eps=1e-5):
    """Generate samples from the model."""
    if self.sampler == 'llada':
      return self._sample_llada(num_samples, num_steps, eps)

    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self.prior_sample(num_samples, self.num_tokens)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in trange(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ancestral':
        _, x = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=None)
      elif self.sampler == 'ancestral_cache':
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=t, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x=x,t=t, dt=dt)

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                    device=self.device)
    if self.config.sampling.noise_removal == 'ancestral':
      if self.sampler == 'analytic':
        x = self._denoiser_update(x=x, t=t0)
      else:
        _, x = self._ancestral_update(x=x, t=t0, dt=None,
                                 p_x0=p_x0_cache,
                                 noise_removal_step=True)
    elif self.config.sampling.noise_removal == 'greedy':
      sigma = self._sigma_from_alphat(self.noise(t0)[1])
      x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
    return x

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, stride_length, num_strides, dt=0.001):
    # TODO(subham): Test this method after refactoring.
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self.prior_sample(n_samples, self.num_tokens)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ancestral_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    # TODO(subham): Test this method after refactoring.
    self._eval_mode()
    (sampling_steps, samples,
     sequence_lengths) = self._semi_ar_sampler(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    self._train_mode()
    return sampling_steps, samples, sequence_lengths


class AbsorbingState(Diffusion):
  def __init__(self, config, tokenizer):
    # NOTE: Ideally, we should do 
    # vocab_size = len(tokenizer), so that we account
    # for the special tokens added in dataloader.py.
    # But we use tokenizer.vocab_size so as to to be
    # consistent with the prior checkpoints.
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    self.subs_masking = config.algo.subs_masking
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    assert not (self.parameterization == 'mean'
                and self.T == 0)
    if self.T > 0:
      assert self.parameterization in {'mean', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'mean'

  def _get_sensitivity_config(self):
    sensitivity_cfg = getattr(self.config.eval, 'sensitivity', None)
    if sensitivity_cfg is None:
      return None
    if not getattr(sensitivity_cfg, 'enabled', False):
      return None
    if self.parameterization != 'subs':
      return None
    return sensitivity_cfg

  def _get_off_trajectory_reg_config(self):
    reg_cfg = getattr(self.config.algo, 'off_trajectory_reg', None)
    if reg_cfg is None:
      return None
    if not getattr(reg_cfg, 'enabled', False):
      return None
    if float(getattr(reg_cfg, 'weight', 0.0)) <= 0:
      return None
    if self.parameterization != 'subs':
      return None
    return reg_cfg

  def _sample_regularizer_time_points(self, reg_cfg):
    time_points = [
      float(time_value)
      for time_value in getattr(reg_cfg, 'time_points', [])
    ]
    if not time_points:
      return []
    num_points = int(
      getattr(reg_cfg, 'num_time_points_per_batch', 1))
    if num_points >= len(time_points):
      return time_points
    permutation = torch.randperm(
      len(time_points), device=self.device)[:num_points]
    return [time_points[int(i)] for i in permutation.cpu().tolist()]

  def _compute_off_trajectory_regularizer(self, x0):
    reg_cfg = self._get_off_trajectory_reg_config()
    if reg_cfg is None:
      return None, None

    delta_t = float(getattr(reg_cfg, 'delta_t', 0.0))
    if delta_t <= 0:
      return None, None

    reg_losses = []
    kept_mask_fracs = []
    selected_time_points = self._sample_regularizer_time_points(
      reg_cfg)
    if not selected_time_points:
      return None, None

    for time_value in selected_time_points:
      t_value = float(time_value)
      if t_value <= self.sampling_eps:
        continue
      t = torch.full(
        (x0.shape[0],),
        t_value,
        device=x0.device,
        dtype=torch.float32)
      s = (t - delta_t).clamp(min=self.sampling_eps)
      effective_delta_t = float((t[0] - s[0]).item())
      if effective_delta_t <= 0:
        continue

      _, alpha_t = self.noise(t)
      _, alpha_s = self.noise(s)
      alpha_t = alpha_t.unsqueeze(-1)
      alpha_s = alpha_s.unsqueeze(-1)
      sigma_t = self._sigma_from_alphat(alpha_t)
      sigma_s = self._sigma_from_alphat(alpha_s)
      xt = self.q_xt(x0, alpha_t)

      with torch.no_grad():
        log_p_t = self.forward(xt, sigma_t)
        _, xs_self = self._ancestral_update(
          x=xt,
          t=t.unsqueeze(-1),
          dt=effective_delta_t,
          p_x0=log_p_t.exp())

      self_mask = (xs_self == self.mask_index)
      if self.ignore_bos:
        self_mask[:, 0] = False
      num_self_masked = self_mask.sum()
      if num_self_masked.item() == 0:
        continue

      log_p_self = self.forward(xs_self, sigma_s)
      self_positions = self_mask.nonzero(as_tuple=False)
      reg_loss = self._target_nll_on_positions(
        log_p_self, x0, self_positions).mean()
      reg_losses.append(reg_loss)
      kept_mask_fracs.append(
        num_self_masked.float() / self_mask.numel())

    if not reg_losses:
      return None, None

    return (
      torch.stack(reg_losses).mean(),
      torch.stack(kept_mask_fracs).mean())

  def _on_validation_epoch_start_extra(self):
    sensitivity_cfg = self._get_sensitivity_config()
    self._sensitivity_metrics = {}
    if sensitivity_cfg is None:
      return
    metric_names = []
    if getattr(sensitivity_cfg, 'pathwise_enabled', True):
      metric_names.extend([
        'val/sensitivity_tv',
        'val/sensitivity_js',
        'val/path_sensitivity_tv',
        'val/path_sensitivity_js',
        'val/path_reveal_frac',
        'val/path_persistent_frac',
        'val/oracle_gap_nll',
        'val/oracle_self_nll',
        'val/oracle_reference_nll',
      ])
    if getattr(sensitivity_cfg, 'flip_enabled', True):
      metric_names.extend([
        'val/flip_sensitivity_tv',
        'val/flip_sensitivity_js',
        'val/flip_reveal_frac',
      ])
    for metric_name in metric_names:
      self._sensitivity_metrics[metric_name] = (
        torchmetrics.aggregation.MeanMetric().to(
          self.device))
    if getattr(sensitivity_cfg, 'log_per_timepoint', False):
      for time_value in sensitivity_cfg.time_points:
        time_label = self._format_sensitivity_time_label(
          time_value)
        for metric_name in metric_names:
          self._sensitivity_metrics[
            f'{metric_name}_{time_label}'] = (
              torchmetrics.aggregation.MeanMetric().to(
                self.device))

  def _on_validation_step_extra(self, batch, batch_idx):
    sensitivity_cfg = self._get_sensitivity_config()
    if sensitivity_cfg is None:
      return
    if batch_idx >= int(sensitivity_cfg.num_batches):
      return
    self._record_sensitivity_proxies(batch['input_ids'])

  def _on_validation_epoch_end_extra(self):
    if not getattr(self, '_sensitivity_metrics', None):
      return
    for metric_name, metric in self._sensitivity_metrics.items():
      metric_weight = getattr(metric, 'weight', None)
      if metric_weight is not None:
        if metric_weight.detach().float().item() <= 0:
          continue
      self.log(metric_name,
               metric.compute(),
               on_epoch=True,
               on_step=False,
               sync_dist=True)

  def _format_sensitivity_time_label(self, time_value):
    return f't{int(round(float(time_value) * 100)):02d}'

  @torch.no_grad()
  def _record_sensitivity_proxies(self, x0):
    sensitivity_cfg = self._get_sensitivity_config()
    if sensitivity_cfg is None:
      return

    delta_t = float(sensitivity_cfg.delta_t)
    pathwise_enabled = bool(
      getattr(sensitivity_cfg, 'pathwise_enabled', True))
    flip_enabled = bool(
      getattr(sensitivity_cfg, 'flip_enabled', True))
    if not pathwise_enabled and not flip_enabled:
      return

    for time_value in sensitivity_cfg.time_points:
      t_value = float(time_value)
      if t_value <= self.sampling_eps:
        continue
      t = torch.full(
        (x0.shape[0],),
        t_value,
        device=x0.device,
        dtype=torch.float32)
      s = (t - delta_t).clamp(min=self.sampling_eps)
      effective_delta_t = float((t[0] - s[0]).item())
      if effective_delta_t <= 0:
        continue

      _, alpha_t = self.noise(t)
      _, alpha_s = self.noise(s)
      alpha_t = alpha_t.unsqueeze(-1)
      alpha_s = alpha_s.unsqueeze(-1)
      xt = self.q_xt(x0, alpha_t)

      sigma_t = self._sigma_from_alphat(alpha_t)
      log_p_t = self.forward(xt, sigma_t)
      time_label = self._format_sensitivity_time_label(
        time_value)
      if pathwise_enabled:
        sigma_s = self._sigma_from_alphat(alpha_s)
        self._record_pathwise_sensitivity(
          x0=x0,
          xt=xt,
          log_p_t=log_p_t,
          alpha_t=alpha_t,
          alpha_s=alpha_s,
          sigma_s=sigma_s,
          t=t.unsqueeze(-1),
          dt=effective_delta_t,
          time_label=time_label,
          sensitivity_cfg=sensitivity_cfg)
      if flip_enabled:
        self._record_flip_sensitivity(
          xt=xt,
          log_p_t=log_p_t,
          sigma_t=sigma_t,
          time_label=time_label,
          sensitivity_cfg=sensitivity_cfg)

  def _update_sensitivity_metric(self, metric_name, value,
                                 time_label=None):
    metric = self._sensitivity_metrics.get(metric_name)
    if metric is not None:
      metric.update(value)
    if time_label is None:
      return
    timed_metric = self._sensitivity_metrics.get(
      f'{metric_name}_{time_label}')
    if timed_metric is not None:
      timed_metric.update(value)

  def _sample_positions(self, positions, max_positions):
    if positions.numel() == 0:
      return positions
    if positions.shape[0] <= max_positions:
      return positions
    selection = torch.randperm(
      positions.shape[0],
      device=positions.device)[:max_positions]
    return positions[selection]

  def _compute_distribution_drift(self, log_p_ref, log_p_other):
    log_p_ref = log_p_ref.float()
    log_p_other = log_p_other.float()
    p_ref = log_p_ref.exp()
    p_other = log_p_other.exp()
    tv = 0.5 * (p_ref - p_other).abs().sum(dim=-1)
    mean_prob = (0.5 * (p_ref + p_other)).clamp_min(1e-12)
    js = 0.5 * (
      F.kl_div(
        log_p_ref,
        mean_prob,
        reduction='none').sum(dim=-1)
      + F.kl_div(
        log_p_other,
        mean_prob,
        reduction='none').sum(dim=-1))
    return tv, js

  def _target_nll_on_positions(self, log_p, x0, positions):
    row_index = positions[:, 0]
    col_index = positions[:, 1]
    targets = x0[row_index, col_index]
    return -log_p[row_index, col_index, targets].float()

  def _sample_coupled_oracle_state(self, x0, xt, alpha_t, alpha_s):
    masked_mask = (xt == self.mask_index)
    denom = (1 - alpha_t).clamp_min(1e-8)
    reveal_prob = ((alpha_s - alpha_t) / denom).clamp(
      min=0.0, max=1.0)
    reveal_draw = (
      torch.rand_like(
        xt, dtype=torch.float32) < reveal_prob.expand_as(xt))
    reveal_mask = masked_mask & reveal_draw
    xs_oracle = xt.clone()
    xs_oracle[reveal_mask] = x0[reveal_mask]
    return xs_oracle

  def _record_pathwise_sensitivity(self, x0, xt, log_p_t,
                                   alpha_t, alpha_s, sigma_s,
                                   t, dt, time_label,
                                   sensitivity_cfg):
    _, xs = self._ancestral_update(
      x=xt,
      t=t,
      dt=dt,
      p_x0=log_p_t.exp())
    masked_mask = (xt == self.mask_index)
    persistent_mask = masked_mask & (xs == self.mask_index)
    masked_count = masked_mask.sum()
    if masked_count.item() == 0:
      return
    persistent_count = persistent_mask.sum()
    revealed_count = masked_count - persistent_count
    masked_count = masked_count.float()
    self._update_sensitivity_metric(
      'val/path_reveal_frac',
      revealed_count.float() / masked_count,
      time_label=time_label)
    self._update_sensitivity_metric(
      'val/path_persistent_frac',
      persistent_count.float() / masked_count,
      time_label=time_label)

    persistent_positions = persistent_mask.nonzero(
      as_tuple=False)
    if persistent_positions.numel() == 0:
      return
    persistent_positions = self._sample_positions(
      persistent_positions,
      int(sensitivity_cfg.max_masked_positions))
    row_index = persistent_positions[:, 0]
    col_index = persistent_positions[:, 1]
    log_p_s = self.forward(xs, sigma_s)
    log_p_ref = log_p_t[row_index, col_index]
    log_p_next = log_p_s[row_index, col_index]
    tv, js = self._compute_distribution_drift(
      log_p_ref, log_p_next)
    tv_rate = tv / dt
    self._update_sensitivity_metric(
      'val/sensitivity_tv',
      tv_rate,
      time_label=time_label)
    self._update_sensitivity_metric(
      'val/sensitivity_js',
      js,
      time_label=time_label)
    self._update_sensitivity_metric(
      'val/path_sensitivity_tv',
      tv_rate,
      time_label=time_label)
    self._update_sensitivity_metric(
      'val/path_sensitivity_js',
      js,
      time_label=time_label)

    xs_oracle = self._sample_coupled_oracle_state(
      x0=x0,
      xt=xt,
      alpha_t=alpha_t,
      alpha_s=alpha_s)
    oracle_common_mask = (
      masked_mask
      & persistent_mask
      & (xs_oracle == self.mask_index))
    oracle_positions = oracle_common_mask.nonzero(
      as_tuple=False)
    if oracle_positions.numel() == 0:
      return
    oracle_positions = self._sample_positions(
      oracle_positions,
      int(sensitivity_cfg.max_masked_positions))
    log_p_oracle = self.forward(xs_oracle, sigma_s)
    oracle_self_nll = self._target_nll_on_positions(
      log_p_s, x0, oracle_positions)
    oracle_ref_nll = self._target_nll_on_positions(
      log_p_oracle, x0, oracle_positions)
    self._update_sensitivity_metric(
      'val/oracle_gap_nll',
      oracle_self_nll - oracle_ref_nll,
      time_label=time_label)
    self._update_sensitivity_metric(
      'val/oracle_self_nll',
      oracle_self_nll,
      time_label=time_label)
    self._update_sensitivity_metric(
      'val/oracle_reference_nll',
      oracle_ref_nll,
      time_label=time_label)

  def _record_flip_sensitivity(self, xt, log_p_t, sigma_t,
                               time_label, sensitivity_cfg):
    masked_positions = (xt == self.mask_index).nonzero(
      as_tuple=False)
    if masked_positions.numel() == 0:
      return
    masked_count = masked_positions.shape[0]
    flip_fraction = float(
      getattr(sensitivity_cfg, 'flip_fraction', 0.02))
    max_flip_positions = int(
      getattr(sensitivity_cfg, 'max_flip_positions', 16))
    num_flip_positions = min(
      max_flip_positions,
      max(1, int(np.ceil(masked_count * flip_fraction))))
    masked_confidence = log_p_t.exp().amax(dim=-1)[
      xt == self.mask_index]
    selection = masked_confidence.topk(
      k=num_flip_positions,
      sorted=False).indices
    flip_positions = masked_positions[selection]
    x_flip = xt.clone()
    predicted_tokens = log_p_t.argmax(dim=-1)
    row_index = flip_positions[:, 0]
    col_index = flip_positions[:, 1]
    x_flip[row_index, col_index] = predicted_tokens[
      row_index, col_index]
    self._update_sensitivity_metric(
      'val/flip_reveal_frac',
      torch.tensor(
        num_flip_positions / masked_count,
        device=xt.device,
        dtype=torch.float32),
      time_label=time_label)

    remaining_mask = ((xt == self.mask_index)
                      & (x_flip == self.mask_index))
    remaining_positions = remaining_mask.nonzero(
      as_tuple=False)
    if remaining_positions.numel() == 0:
      return
    remaining_positions = self._sample_positions(
      remaining_positions,
      int(sensitivity_cfg.max_masked_positions))
    log_p_flip = self.forward(x_flip, sigma_t)
    row_index = remaining_positions[:, 0]
    col_index = remaining_positions[:, 1]
    log_p_ref = log_p_t[row_index, col_index]
    log_p_other = log_p_flip[row_index, col_index]
    tv, js = self._compute_distribution_drift(
      log_p_ref, log_p_other)
    self._update_sensitivity_metric(
      'val/flip_sensitivity_tv',
      tv,
      time_label=time_label)
    self._update_sensitivity_metric(
      'val/flip_sensitivity_js',
      js,
      time_label=time_label)

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False):
    losses = super()._loss(
      x0=x0,
      valid_tokens=valid_tokens,
      current_accumulation_step=current_accumulation_step,
      train_mode=train_mode)
    reg_cfg = self._get_off_trajectory_reg_config()
    if (not train_mode or reg_cfg is None):
      return losses

    reg_loss, kept_mask_frac = self._compute_off_trajectory_regularizer(
      x0=x0)
    if reg_loss is None:
      return losses

    reg_weight = float(getattr(reg_cfg, 'weight', 0.0))
    weighted_reg_loss = reg_weight * reg_loss
    self.log('train/off_trajectory_reg_loss',
             reg_loss.detach(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    self.log('train/off_trajectory_reg_weighted',
             weighted_reg_loss.detach(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    if kept_mask_frac is not None:
      self.log('train/off_trajectory_mask_frac',
               kept_mask_frac.detach(),
               on_step=True,
               on_epoch=False,
               sync_dist=True)
    return Loss(
      loss=losses.loss + weighted_reg_loss,
      nlls=losses.nlls,
      prior_loss=losses.prior_loss,
      num_tokens=losses.num_tokens)

  def q_xt(self, x, alpha_t):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      alpha_t: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < 1 - alpha_t
    xt = torch.where(move_indices, self.mask_index, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      p_x0 = self.forward(
        x, self._sigma_from_alphat(alpha_t)).exp()

    p_x0_update = p_x0
    x0_temperature = float(getattr(
      self.config.sampling, 'x0_temperature', 1.0))
    if x0_temperature > 0 and x0_temperature != 1.0:
      x0_temperature_schedule = getattr(
        self.config.sampling, 'x0_temperature_schedule', 'constant')
      # Schedule shape parameter for power/cosine/budget schedules.
      # Larger -> sharpening more localized at high noise (alpha_t close to 1).
      schedule_power = float(getattr(
        self.config.sampling, 'x0_temperature_schedule_power', 1.0))
      a = alpha_t[:, :, None].clamp(0.0, 1.0)
      if x0_temperature_schedule == 'constant':
        step_temperature = x0_temperature
      elif x0_temperature_schedule == 'anneal_to_one':
        # linear in alpha_t: tau at alpha=0, 1 at alpha=1
        step_temperature = x0_temperature + (
          1 - x0_temperature) * a
      elif x0_temperature_schedule == 'anneal_from_one':
        # linear in alpha_t: 1 at alpha=0, tau at alpha=1
        step_temperature = 1 + (
          x0_temperature - 1) * a
      elif x0_temperature_schedule == 'power_alpha':
        # tau(t) = tau_base ** (alpha_t ** p). p=1 reduces to anneal_from_one in log-space.
        # p<1 spreads sharpening toward low noise; p>1 concentrates at high noise.
        log_tau = torch.log(torch.tensor(
          x0_temperature, device=a.device, dtype=a.dtype))
        step_temperature = torch.exp(log_tau * (a ** schedule_power))
      elif x0_temperature_schedule == 'cosine_alpha':
        # smooth cosine ramp in alpha_t: tau at alpha=1, 1 at alpha=0
        ramp = 0.5 * (1 - torch.cos(torch.pi * a))
        step_temperature = 1 + (x0_temperature - 1) * ramp
      elif x0_temperature_schedule == 'budget_aware':
        # tau_eff(t) = 1 + (tau - 1) * (alpha_t - alpha_s)/(1 - alpha_s)
        # i.e. sharpening proportional to mass of tokens revealed in this step,
        # which is the theoretical commitment-risk weighting (1/N at uniform grids).
        a_s = alpha_s[:, :, None].clamp(0.0, 1.0)
        weight = ((a - a_s) / (1 - a_s).clamp_min(1e-6)).clamp(0.0, 1.0)
        step_temperature = 1 + (x0_temperature - 1) * weight
      else:
        raise ValueError(
          f'Unknown x0 temperature schedule: {x0_temperature_schedule}')
      p_x0_update = (
        p_x0.clamp_min(1e-30).log() / step_temperature).softmax(dim=-1)

    reveal_strategy = getattr(
      self.config.sampling, 'ancestral_reveal_strategy', 'independent')
    if reveal_strategy in {'confidence_topk', 'topk_confidence'}:
      if noise_removal_step:
        reveal_strategy = 'independent'
      else:
        masked = x == self.mask_index
        reveal_prob = ((alpha_s - alpha_t) / (1 - alpha_t)).clamp(0, 1)
        expected_reveals = masked.sum(dim=-1).to(reveal_prob.dtype) * reveal_prob.squeeze(-1)
        num_reveals = expected_reveals.floor().to(torch.long)
        fractional = expected_reveals - num_reveals
        num_reveals = num_reveals + (
          torch.rand_like(fractional) < fractional).to(torch.long)
        num_reveals = torch.minimum(num_reveals, masked.sum(dim=-1))

        x0_samples = sample_categorical(
          p_x0_update, precision=self.config.sampling.gumbel_precision)
        confidence = p_x0_update.max(dim=-1).values
        scores = confidence.masked_fill(~masked, -torch.inf)
        max_reveals = int(num_reveals.max().item())
        reveal_mask = torch.zeros_like(masked)
        if max_reveals > 0:
          top_indices = scores.topk(max_reveals, dim=-1).indices
          ranks = torch.arange(max_reveals, device=x.device).unsqueeze(0)
          selected = ranks < num_reveals.unsqueeze(-1)
          reveal_mask.scatter_(1, top_indices, selected)
        return p_x0, torch.where(reveal_mask, x0_samples, x)
    elif reveal_strategy != 'independent':
      raise ValueError(
        f'Unknown ancestral reveal strategy: {reveal_strategy}')
    
    q_xs = p_x0_update * (alpha_s - alpha_t)[:, :, None]
    q_xs[:, :, self.mask_index] = 1 - alpha_s
    _x = sample_categorical(q_xs, precision=self.config.sampling.gumbel_precision)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, dt):
    sigma_t = self._sigma_from_alphat(self.noise(t)[1])
    sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
    dsigma = sigma_t - sigma_s
    score = self._get_score(x, sigma_t)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return sample_categorical(probs, precision=self.config.sampling.gumbel_precision)

  def _denoiser_update(self, x, t):
    sigma = self._sigma_from_alphat(self.noise(t)[1])
    score = self._get_score(x, sigma)
    if self.config.sampling.use_float64:
      score = score.to(torch.float64)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = sample_categorical(probs, precision=self.config.sampling.gumbel_precision)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _get_num_transfer_tokens(self, mask_index, steps):
    """Calculate number of tokens to transfer at each step for current block."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens

  def _llada_update(self, x, t, mask_region, num_transfer_tokens, temperature, remasking_strategy, threshold):
    """LLaDA-style update with confidence threshold support."""
    precision = self.config.sampling.gumbel_precision

    # Get model predictions
    _, alpha_t = self.noise(t)
    assert alpha_t.ndim == 2
    unet_conditioning = self._sigma_from_alphat(alpha_t)
    logits = self.forward(x, unet_conditioning)
    
    # Apply Gumbel noise for sampling
    ## won't use sample_categorical here
    if temperature > 0:
        precision_dtype = _sampling_dtype(precision)
        noise = torch.rand_like(logits, dtype=precision_dtype)
        noise = noise.clamp_min(torch.finfo(precision_dtype).tiny)
        gumbel_noise = (-torch.log(noise)) ** temperature
        logits_with_noise = logits.to(precision_dtype).exp() / gumbel_noise
        x0 = torch.argmax(logits_with_noise, dim=-1)
    else:
        x0 = torch.argmax(logits, dim=-1)
    
    # Calculate confidence scores
    if remasking_strategy == 'low_confidence':
        ### diversity handling???
        # if threshold != -1 and temperature > 0:
        #   logits = logits / temperature
        ###

        p = F.softmax(logits.to(_sampling_dtype(precision)), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking_strategy == 'random':
        x0_p = torch.rand_like(x0, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")
    
    # Only consider masked positions in the target region
    x0 = torch.where(mask_region, x0, x)
    ### also introduce a noisy confidence version??? that is built on logits_noise
    confidence = torch.where(mask_region, x0_p, -float('inf'))
    
    # Select tokens to transfer
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    
    for j in range(confidence.shape[0]):
        valid_positions = mask_region[j].sum().item()
        if valid_positions == 0:
            continue

        if threshold != -1:
            # Use all valid positions, then filter by threshold
            k = valid_positions
        else:
            # Use scheduled number
            k = min(num_transfer_tokens[j].item(), valid_positions)
        
        if k > 0:
            _, select_index = torch.topk(confidence[j], k=k)

            if threshold != -1:
                # Filter by threshold
                _, tmp_index = torch.topk(confidence[j], k=5)
                print(confidence[j, tmp_index])
                valid_mask = confidence[j, select_index] >= threshold

                ### Preliminary handling for no valid tokens
                # if valid_mask.sum() == 0:
                #     print(f"No tokens above threshold {threshold} in block {j + 1}.")
                #     k = min(num_transfer_tokens[j].item(), valid_positions)
                #     _, select_index = torch.topk(confidence[j], k=k)
                # else:
                select_index = select_index[valid_mask]
                print(f"Selected {len(select_index)} tokens for transfer in block {j + 1} with threshold {threshold}.")
            
            if len(select_index) > 0:
                transfer_index[j, select_index] = True
    
    # Apply updates
    x_new = x.clone()
    x_new[transfer_index] = x0[transfer_index]
    return x_new

  @torch.no_grad()
  def _sample_llada(self, num_samples, num_steps=None, eps=1e-5):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self.prior_sample(num_samples, self.num_tokens).to(self.device)

    ### Prompt handling (pending transfer)
    ###

    block_length = self.config.sampling.block_length
    assert self.num_tokens % block_length == 0
    num_blocks = self.num_tokens // block_length
    # Distribute steps across blocks
    assert num_steps % num_blocks == 0
    steps_per_block = num_steps // num_blocks
    
    # Fixed timestep for LLaDA (low noise)
    ### Examine default value
    t_fixed = getattr(self.config.sampling, 'llada_t_fixed', eps)
    t = t_fixed * torch.ones(x.shape[0], 1, device=self.device)

    # Temperature scheduling
    temp_start = getattr(self.config.sampling, 'temperature_start', self.config.sampling.temperature)
    temp_end = getattr(self.config.sampling, 'temperature_end', self.config.sampling.temperature)
    use_temp_schedule = temp_start != temp_end
    
    for block_idx in range(num_blocks):
      print(f"Processing block {block_idx + 1}/{num_blocks}...")
      # unlike llada official implementation, I do not worry about prompt length and shifting block start
      # it's handled naturally due to being unmasked since start
      block_start = block_idx * block_length
      block_end = block_start + block_length
      
      # Get initial masks in current block
      block_mask = (x[:, block_start:block_end] == self.mask_index)
      if block_mask.sum() == 0:
        # No masks in this block (e.g., prompt region); skip
        continue
      
      # Calculate transfer schedule for this block
      num_transfer_schedule = self._get_num_transfer_tokens(block_mask, steps_per_block)
      
      # Iterative refinement within block
      for step_idx in trange(steps_per_block):
        # Calculate temperature for this step
        if use_temp_schedule:
          print(f"Using temperature schedule: {temp_start} -> {temp_end}")
          # Linear decay across all blocks and steps
          total_steps = num_blocks * steps_per_block
          current_step = block_idx * steps_per_block + step_idx
          progress = current_step / (total_steps - 1) if total_steps > 1 else 1.0
          
          if self.config.sampling.temperature_schedule == 'lin':
            # Linear decay
            temperature_curr = temp_start + (temp_end - temp_start) * progress
          elif self.config.sampling.temperature_schedule == 'cos':
            # Cosine decay (smoother transition)
            temperature_curr = temp_end + (temp_start - temp_end) * 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
          elif self.config.sampling.temperature_schedule == 'exp':
            # Exponential decay (faster initial drop)
            decay_rate = torch.log(torch.tensor(temp_end / temp_start))
            temperature_curr = temp_start * torch.exp(decay_rate * progress)
          elif self.config.sampling.temperature_schedule == 'fastexp':
            # Exponential decay (faster initial drop)
            decay_rate = np.log(temp_end / temp_start)
            temperature_curr = temp_start * np.exp(decay_rate * progress)
          else:
            raise ValueError(f"Unknown temperature schedule: {self.config.sampling.temperature_schedule}")

          print(f"Current temperature: {temperature_curr}")
        else:
          temperature_curr = self.config.sampling.temperature


        # Define mask region (only current block)
        mask_region = torch.zeros_like(x, dtype=torch.bool)
        current_masks = (x[:, block_start:block_end] == self.mask_index)
        mask_region[:, block_start:block_end] = current_masks
        
        if mask_region.sum() == 0:
          # No more masks in this block
          break
        
        x = self._llada_update(
          x=x,
          t=t,
          mask_region=mask_region,
          num_transfer_tokens=num_transfer_schedule[:, step_idx],
          temperature=temperature_curr,
          remasking_strategy=self.config.sampling.remasking_strategy,
          threshold=getattr(self.config.sampling, 'llada_threshold', -1)
        )
        
        # Check if block is complete
        if (x[:, block_start:block_end] == self.mask_index).sum() == 0:
          break

    ### Noise removal (REMOVE???)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
    if self.config.sampling.noise_removal == 'ancestral':
      _, x = self._ancestral_update(x=x, t=t0, dt=None,
                                p_x0=None,
                                noise_removal_step=True)
    elif self.config.sampling.noise_removal == 'greedy':
      sigma = self._sigma_from_alphat(self.noise(t0)[1])
      x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
    return x


class UniformState(Diffusion):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.time_conditioning
    assert self.parameterization == 'mean'
    if self.config.algo.name != 'distillation':
      assert self.T == 0

  def q_xt(self, x, alpha_t):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      move_chance: float torch.Tensor with shape
        (batch_size, 1).
    """
    move_indices = torch.rand(
      *x.shape, device=x.device) < 1 - alpha_t
    uniform_tensor = torch.randint(
      0, self.vocab_size, x.shape, device=x.device)
    xt = torch.where(move_indices, uniform_tensor, x)
    if self.ignore_bos:
      xt[:, 0] = x[:, 0]
    return xt

  def prior_sample(self, *batch_dims):
    return torch.randint(
      0, self.vocab_size, batch_dims, dtype=torch.int64,
      device=self.device)
