import os
import collections
import copy
import pickle

import fsspec
import math
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning
    assert self.config.prior.type == 'none'

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  # def nll(self, input_tokens, output_tokens,
  #         current_accumulation_step):
  #   del current_accumulation_step
  def nll(self, input_tokens, output_tokens,
          current_accumulation_step, train_mode=False):
    del current_accumulation_step
    del train_mode
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    # ancestral sampling isn't desirable because it's slow
    assert self.sampler in ['ancestral_cache', 'llada']

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del low_var
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    if (self.T > 0
        and getattr(self.config.algo,
                    'discrete_loss_exact_mask_norm',
                    True)):
      # Match the discrete-budget objective more closely by averaging the
      # masked-token cross-entropy over the realized masked set rather than
      # using the continuous-time 1 / t importance weight.
      mask = (xt == self.mask_index).to(log_p_theta.dtype)
      num_masked = mask.sum(dim=-1, keepdim=True).clamp(min=1)
      return (-log_p_theta) * mask / num_masked * xt.shape[1]
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class CDLM(trainer_base.AbsorbingState):
    """
    CDLM: Consistent Diffusion Language Model (masked absorbing-state variant).

    - Trains with a consistency objective between a student (current model)
      and a teacher (EMA target) at two nearby noise levels s<t (Δt=t-s).
    - Supports multiple Δt schedules (constant, staged, linear inc/dec).
    - When min_delta_t == -1, switches to MDLM behavior (pure diffusion loss).

    Expected config.algo keys (commonly used here):
      - consistency_weight: float, extra weight on consistency term
      - target_ema_rate: float in [0,1), EMA for target network
      - delta_t_schedule: one of {"constant", "staged_decrease", "staged_increase",
                                 "linear_increase", "linear_decrease"}
      - min_delta_t, max_delta_t: bounds for Δt
      - update_ema_every, ema_warmup_steps: EMA cadence/warmup
      - loss_type: {"kl-fwd", "kl-bwd", "cross_entropy", "js", "ejs", "ejs0", "topk-<k>-<base>"}
      - max_weight_clip, eps_stability, loss_clip
      - use_diffusion_loss, diffusion_loss_weight
      - use_reconstruction_loss, reconstruction_loss_weight
      - use_fixed_grid: bool, optional (default False)
      - finetune_total_steps: for linear schedules
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        # Core hyperparameters
        self.consistency_weight = getattr(config.algo, "consistency_weight", 1.0)
        self.target_ema_rate = getattr(config.algo, "target_ema_rate", 0.999)
        self.delta_t_schedule = getattr(config.algo, "delta_t_schedule", "constant")
        self.min_delta_t = getattr(config.algo, "min_delta_t", 0.001)
        self.max_delta_t = getattr(config.algo, "max_delta_t", 0.10)
        self.loss_type = getattr(config.algo, "loss_type", "kl-fwd")
        self.update_ema_every = getattr(config.algo, "update_ema_every", 1)
        
        # Numerical stability/safety
        self.max_weight_clip = getattr(config.algo, "max_weight_clip", 100.0)
        self.eps_stability = getattr(config.algo, "eps_stability", 1e-6)
        self.loss_clip = getattr(config.algo, "loss_clip", 1000.0)

        self._validate_configuration()

        # EMA teacher network for consistency training
        if self.target_ema_rate > 0:
            self._init_target_network()

        # Warmup: copy weights directly during early training
        self.ema_warmup_steps = getattr(config.algo, "ema_warmup_steps", 20000)

        # Fine-tuning bookkeeping (used for linear schedules)
        self.finetune_start_step = None
        self.finetune_total_steps = getattr(config.algo, "finetune_total_steps", 10000)

        # Logging helpers
        self.last_training_delta_t = None
        self._last_diffusion_loss = None

    def _validate_configuration(self):
        # CDLM operates with mean/score parameterizations and supported samplers
        assert self.parameterization in ["mean", "score"]
        assert self.sampler in ['ancestral', 'ancestral_cache', 'llada', 'analytic']

    def _init_target_network(self):
        """Initialize the EMA teacher as a frozen copy of the backbone."""
        import copy
        self.target_backbone = copy.deepcopy(self.backbone)
        for p in self.target_backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self):
        """EMA update for the teacher (full copy during warmup)."""
        if self.target_ema_rate <= 0:
            return
        if self.global_step < self.ema_warmup_steps:
            for p, tp in zip(self.backbone.parameters(), self.target_backbone.parameters()):
                tp.data.copy_(p.data)
        else:
            for p, tp in zip(self.backbone.parameters(), self.target_backbone.parameters()):
                tp.data.mul_(self.target_ema_rate).add_(p.data, alpha=1 - self.target_ema_rate)

    def optimizer_step(self, *args, **kwargs):
        """Optimizer step with optional grad clipping and EMA update."""
        if hasattr(self, 'trainer') and self.trainer is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        super().optimizer_step(*args, **kwargs)
        if self.target_ema_rate > 0 and self.global_step % self.update_ema_every == 0:
            self._update_target_network()

    def _get_current_delta_t(self, current_iteration=None):
        """Return Δt>0 (≤1-sampling_eps) based on schedule. Set min_delta_t=-1 for MDLM mode."""
        if self.min_delta_t == -1:
            return -1

        # Predefined staged schedules (kept for backwards-compat experiments)
        stages_09001 = [(0, 0.9), (2000, 0.8), (4000, 0.7), (6000, 0.6), (8000, 0.5), (10000, 0.4), (12000, 0.3), (14000, 0.2), (16000, 0.1), (18000, 0.05), (20000, 0.01), (22000, 0.001)]
        stages_05001 = [(0, 0.5), (3000, 0.4), (6000, 0.3), (9000, 0.2), (12000, 0.1), (15000, 0.05), (18000, 0.01)]
        stages_00105 = [(0, 0.01), (3000, 0.05), (6000, 0.1), (9000, 0.2), (12000, 0.3), (15000, 0.4), (18000, 0.5)]

        if self.delta_t_schedule == "constant":
            delta_t = self.min_delta_t
        elif self.delta_t_schedule == "staged_decrease":
            stages = stages_09001  # choose default staged decrease ladder
            delta_t = self.min_delta_t
            if current_iteration is not None:
                for step_threshold, delta_value in stages:
                    if current_iteration >= step_threshold:
                        delta_t = delta_value
        elif self.delta_t_schedule == "staged_increase":
            stages = stages_00105
            delta_t = self.min_delta_t
            if current_iteration is not None:
                for step_threshold, delta_value in stages:
                    if current_iteration >= step_threshold:
                        delta_t = delta_value
        elif self.delta_t_schedule == "linear_increase":
            if current_iteration is None:
                delta_t = self.min_delta_t
            else:
                progress = min(current_iteration / self.finetune_total_steps, 1.0)
                delta_t = self.min_delta_t + progress * (self.max_delta_t - self.min_delta_t)
        elif self.delta_t_schedule == "linear_decrease":
            if current_iteration is None:
                delta_t = self.max_delta_t
            else:
                progress = min(current_iteration / self.finetune_total_steps, 1.0)
                delta_t = self.max_delta_t - progress * (self.max_delta_t - self.min_delta_t)
        else:
            raise ValueError(f"Unknown delta_t_schedule: {self.delta_t_schedule}")
        
        # Ensure Δt never exceeds the final time bound
        delta_t = min(delta_t, 1.0 - self.sampling_eps)
        return delta_t

    def q_xs_given_xt_x0(self, xt, x0, alpha_t, alpha_s):
        """Posterior q(x_s|x_t,x_0) under the absorbing process.
        Returns per-position categorical over V.
        """
        batch_size, seq_len = xt.shape
        q_xs = torch.zeros(batch_size, seq_len, self.vocab_size, 
                          device=xt.device, dtype=torch.float32)
        
        # Case 1: x_t is masked ⇒ can remain masked or reveal x0
        t_masked = (xt == self.mask_index)     
        if t_masked.any():
            alpha_t_safe = alpha_t.clamp(min=self.eps_stability)
            alpha_s_safe = alpha_s.clamp(min=self.eps_stability)
            mask_prob = (1 - alpha_s_safe) / (1 - alpha_t_safe)  # == s/t
            x0_prob = (alpha_s_safe - alpha_t_safe).clamp(min=0) / (1 - alpha_t_safe)
            batch_idx, seq_idx = torch.where(t_masked)
            q_xs[batch_idx, seq_idx, self.mask_index] = mask_prob[batch_idx, 0]
            x0_tokens = x0[batch_idx, seq_idx]
            q_xs[batch_idx, seq_idx, x0_tokens] = x0_prob[batch_idx, 0]
        
        # Case 2: x_t is a real token ⇒ must stay the same
        t_not_masked = ~t_masked
        if t_not_masked.any():
            batch_idx, seq_idx = torch.where(t_not_masked)
            xt_idx = xt[t_not_masked]
            q_xs[batch_idx, seq_idx, xt_idx] = 1.0
        
        # Normalize to defend against rare degenerate rows
        q_xs = q_xs / (q_xs.sum(dim=-1, keepdim=True) + self.eps_stability)
        return q_xs

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=True):
        """Main training objective.
        - MDLM path (min_delta_t==-1): diffusion-only loss.
        - CDLM path: consistency loss between student at t and teacher at s.
        Returns per-position losses (B, L).
        """
        del output_tokens
        
        # Determine Δt at the current iteration
        current_iter = (self.global_step - self.finetune_start_step
                        if self.finetune_start_step is not None else self.global_step)
        delta_t_value = self._get_current_delta_t(current_iter)
        self.last_training_delta_t = delta_t_value

        # Times t≥Δt and s=t-Δt (or MDLM shortcut)
        if delta_t_value == -1:
            t = self._sample_t(x0.shape[0], current_accumulation_step)
            delta_t = t.clone()
            s = torch.zeros_like(t)
        else:
            if self.config.algo.get('use_fixed_grid', False):
                # Use discrete grid t∈{Δt, 2Δt, ..., 1}
                num_steps = int(1.0 / delta_t_value)
                grid_indices = torch.randint(1, num_steps + 1, (x0.shape[0],), device=x0.device)
                t = grid_indices.float() * delta_t_value
                t = t.clamp(max=1.0)
                s = (t - delta_t_value).clamp(min=self.sampling_eps)
                delta_t = torch.full_like(t, delta_t_value)
            else:
                # Uniform t∈[Δt, 1], optionally antithetic
                t = torch.rand(x0.shape[0], device=x0.device) * (1 - delta_t_value) + delta_t_value
                if current_accumulation_step is not None:
                    if self.antithetic_sampling and current_accumulation_step % 2 == 1:
                        t = 1 - t + delta_t_value
                delta_t = torch.full_like(t, delta_t_value)
                s = (t - delta_t).clamp(min=self.sampling_eps)

        # Noise scales and their sigmas
        dalpha_t, alpha_t = self.noise(t)
        dalpha_s, alpha_s = self.noise(s)
        alpha_t = alpha_t.unsqueeze(-1)
        alpha_s = alpha_s.unsqueeze(-1)
        sigma_t = self._sigma_from_alphat(alpha_t)
        sigma_s = self._sigma_from_alphat(alpha_s)

        # Sample xt and xs (exact posterior when Δt>0)
        xt = self.q_xt(x0, alpha_t)
        if delta_t_value == -1:
            xs = x0
        else:
            q_xs_dist = self.q_xs_given_xt_x0(xt, x0, alpha_t, alpha_s)
            xs = trainer_base.sample_categorical(
                q_xs_dist,
                precision=self.config.training.loss_precision
            )
        
        # Forward student at t
        log_p_student = self.forward(xt, sigma=sigma_t)
        
        # (Optional) diffusion loss (MDLM-style)
        if self.config.algo.get('use_diffusion_loss', False):
            log_p_theta_x0 = torch.gather(input=log_p_student, dim=-1, index=x0[:, :, None]).squeeze(-1)
            diffusion_loss = -log_p_theta_x0 / t.unsqueeze(-1).clamp(min=1.0, max=self.max_weight_clip)
            if hasattr(self, 'tokenizer') and self.tokenizer.pad_token_id is not None:
                pad_mask = (x0 != self.tokenizer.pad_token_id).float()
                diffusion_loss = diffusion_loss * pad_mask
            self._last_diffusion_loss = diffusion_loss
        
        # Loss paths
        if delta_t_value == -1:
            # MDLM mode: either return diffusion loss, or uniform teacher heuristic
            if self.config.algo.get('use_diffusion_loss', False):
                consistency_loss = diffusion_loss
            else:
                eps = self.eps_stability
                log_p_teacher = torch.full_like(log_p_student, math.log(eps))
                log_p_teacher.scatter_(-1, x0.unsqueeze(-1), math.log1p(-eps * (self.vocab_size - 1)))
                if self.loss_type == 'kl-bwd':
                    consistency_loss = torch.nn.functional.kl_div(
                        log_p_student, log_p_teacher, reduction='none', log_target=True
                    ).sum(-1)
                elif self.loss_type == 'kl-fwd':
                    consistency_loss = torch.nn.functional.kl_div(
                        log_p_teacher, log_p_student, reduction='none', log_target=True
                    ).sum(-1)
                mask_indices = (xt == self.mask_index).float()
                consistency_loss = consistency_loss * mask_indices
                num_masked = mask_indices.sum(dim=-1, keepdim=True).clamp(min=1)
                consistency_loss = consistency_loss / num_masked * xt.shape[1]
        else:
            # CDLM consistency: student at t vs teacher at s
            with torch.no_grad():
                log_p_teacher = self._forward_target(xs, sigma=sigma_s)
            log_p_teacher = log_p_teacher.clamp_min(math.log(self.eps_stability))
            
            # Choose divergence
            if self.config.training.loss_precision == 'float64':
                log_p_student = log_p_student.to(torch.float64)
                log_p_teacher = log_p_teacher.to(torch.float64)

            if self.loss_type == 'kl-bwd':
                consistency_loss = torch.nn.functional.kl_div(
                    log_p_student, log_p_teacher, reduction='none', log_target=True
                ).sum(-1)
            elif self.loss_type == 'kl-fwd':
                consistency_loss = torch.nn.functional.kl_div(
                    log_p_teacher, log_p_student, reduction='none', log_target=True
                ).sum(-1)
            elif self.loss_type == 'cross_entropy':
                consistency_loss = -torch.sum(log_p_teacher.exp() * log_p_student, dim=-1)
            elif self.loss_type == 'js':
                log_p_avg = torch.logsumexp(torch.stack([log_p_student, log_p_teacher]), dim=0) - math.log(2)
                js_1 = torch.nn.functional.kl_div(log_p_avg, log_p_student, reduction='none', log_target=True).sum(-1)
                js_2 = torch.nn.functional.kl_div(log_p_avg, log_p_teacher, reduction='none', log_target=True).sum(-1)
                consistency_loss = 0.5 * (js_1 + js_2)
            elif self.loss_type == 'ejs0':
                log_p_avg = torch.logaddexp(log_p_student, log_p_teacher) - math.log(2)
                p_avg = log_p_avg.exp()
                consistency_loss = 0.5 * (p_avg * (2 * log_p_avg - log_p_student - log_p_teacher)).sum(-1)
            elif self.loss_type == 'ejs':
                log_p_avg = torch.logaddexp(log_p_student, log_p_teacher) - math.log(2)
                kl_student = torch.nn.functional.kl_div(log_p_avg, log_p_student, reduction='none', log_target=True).sum(-1)
                kl_teacher = torch.nn.functional.kl_div(log_p_avg, log_p_teacher, reduction='none', log_target=True).sum(-1)
                consistency_loss = 0.5 * (kl_student + kl_teacher)
            elif self.loss_type.startswith('topk-'):
                if self.loss_type == 'topk':
                    base_loss_type = 'kl-fwd'
                    k = 10
                else:
                    parts = self.loss_type.split('-')
                    k = int(parts[1]) if len(parts) > 1 else 10
                    base_loss_type = '-'.join(parts[2:]) if len(parts) > 2 else 'kl-bwd'
                _, topk_indices = torch.topk(log_p_teacher, k, dim=-1)
                log_p_student_topk = torch.gather(log_p_student, -1, topk_indices)
                log_p_teacher_topk = torch.gather(log_p_teacher, -1, topk_indices)
                log_p_student_topk = log_p_student_topk - torch.logsumexp(log_p_student_topk, dim=-1, keepdim=True)
                log_p_teacher_topk = log_p_teacher_topk - torch.logsumexp(log_p_teacher_topk, dim=-1, keepdim=True)
                if base_loss_type == 'kl-bwd':
                    consistency_loss = torch.nn.functional.kl_div(
                        log_p_student_topk, log_p_teacher_topk, reduction='none', log_target=True
                    ).sum(-1)
                elif base_loss_type == 'kl-fwd':
                    consistency_loss = torch.nn.functional.kl_div(
                        log_p_teacher_topk, log_p_student_topk, reduction='none', log_target=True
                    ).sum(-1)
                elif base_loss_type == 'cross_entropy':
                    consistency_loss = -torch.sum(log_p_teacher_topk.exp() * log_p_student_topk, dim=-1)
                elif base_loss_type == 'js':
                    log_p_avg_topk = torch.logsumexp(torch.stack([log_p_student_topk, log_p_teacher_topk]), dim=0) - math.log(2)
                    js_1 = torch.nn.functional.kl_div(log_p_avg_topk, log_p_student_topk, reduction='none', log_target=True).sum(-1)
                    js_2 = torch.nn.functional.kl_div(log_p_avg_topk, log_p_teacher_topk, reduction='none', log_target=True).sum(-1)
                    consistency_loss = 0.5 * (js_1 + js_2)
            
            # Weighting, masking, and optional additions
            weight = (1.0 / delta_t.unsqueeze(-1)).clamp(min=1.0, max=self.max_weight_clip)
            consistency_loss = consistency_loss * weight * self.consistency_weight
            mask_indices = (xt == self.mask_index).float()
            consistency_loss = consistency_loss * mask_indices
            num_masked = mask_indices.sum(dim=-1, keepdim=True).clamp(min=1)
            consistency_loss = consistency_loss / num_masked * xt.shape[1]
            
            if self.config.algo.get('use_diffusion_loss', False):
                diff_weight = self.config.algo.get('diffusion_loss_weight', 0.1)
                if train_mode and self.global_step % 100 == 0:
                    self.log('train/consistency_loss_raw', consistency_loss.mean().item(), on_step=True, on_epoch=False, sync_dist=True)
                    self.log('train/diffusion_loss_raw', diffusion_loss.mean().item(), on_step=True, on_epoch=False, sync_dist=True)
                    self.log('train/diffusion_weight', diff_weight, on_step=True, on_epoch=False, sync_dist=True)
                    self.log('train/consistency_to_diffusion_ratio', consistency_loss.mean().item() / (diffusion_loss.mean().item() + 1e-8), on_step=True, on_epoch=False, sync_dist=True)
                consistency_loss = consistency_loss + diff_weight * diffusion_loss

            if self.config.algo.get('use_reconstruction_loss', False):
                rec_weight = self.config.algo.get('reconstruction_loss_weight', 0.1)
                rec_loss = self._reconstruction_loss(x0)
                if train_mode and self.global_step % 100 == 0:
                    self.log('train/reconstruction_weight', rec_weight, on_step=True, on_epoch=False, sync_dist=True)
                    self.log('train/reconstruction_loss_raw', rec_loss.mean().item(), on_step=True, on_epoch=False, sync_dist=True)
                consistency_loss = consistency_loss + rec_weight * rec_loss

        # Final logging
        if train_mode and self.global_step % 100 == 0:
            self.log('train/total_loss', consistency_loss.mean().item(), on_step=True, on_epoch=False, sync_dist=True)
            self.log('train/delta_t_value', delta_t_value, on_step=True, on_epoch=False, sync_dist=True)
        
        if not train_mode:
            self.log('val/total_loss', consistency_loss.mean().item(), on_step=False, on_epoch=True, sync_dist=True)
            if self.config.algo.get('use_diffusion_loss', False) and hasattr(self, '_last_diffusion_loss'):
                self.log('val/diffusion_loss_raw', self._last_diffusion_loss.mean().item(), on_step=False, on_epoch=True, sync_dist=True)
        
        return consistency_loss

    def _process_model_output(self, model_output, xt, sigma):
      """Convert raw logits to log-probs and enforce absorbing constraints."""
      del sigma
      # P(mask | any input) = -inf (can only transition OUT of mask)
      model_output[:, :, self.mask_index] += self.neg_infinity
      
      # Normalize to log-probabilities
      model_output = model_output - torch.logsumexp(model_output, dim=-1, keepdim=True)
      
      # Enforce identity on unmasked tokens (keep observed tokens fixed)
      unmasked_indices = (xt != self.mask_index)
      model_output[unmasked_indices] = self.neg_infinity
      model_output[unmasked_indices, xt[unmasked_indices]] = 0
      
      return model_output

    def _forward_target(self, xt, sigma):
        """Forward pass through the EMA teacher and process outputs."""
        sigma_processed = self._process_sigma(sigma)
        logits = self.target_backbone(xt, sigma_processed)
        return self._process_model_output(logits, xt, sigma)

    def _get_score(self, x, sigma):
        """Return score(x,t) in log-space with absorbing-state handling."""
        model_output = self.forward(x, sigma)
        sigma_clamped = sigma.clamp(min=self.eps_stability)
        log_k = -torch.log(torch.expm1(sigma_clamped)).squeeze(-1)
        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0
        unmasked_score = torch.full_like(model_output, self.neg_infinity)
        unmasked_score = torch.scatter(unmasked_score, -1, x[..., None], torch.zeros_like(unmasked_score[..., :1]))
        unmasked_score[:, :, self.mask_index] = -log_k[:, None].expand_as(x)
        masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
        model_output = masked_score * masked_indices + unmasked_score * (1 - masked_indices)
        return (model_output - model_output.max(dim=-1, keepdim=True)[0]).exp()

    def on_load_checkpoint(self, checkpoint):
        """Load EMA teacher (if present) and record fine-tune start step."""
        super().on_load_checkpoint(checkpoint)
        if self.target_ema_rate > 0 and 'target_backbone_state_dict' in checkpoint:
            self.target_backbone.load_state_dict(checkpoint['target_backbone_state_dict'])
        if self.training and self.finetune_start_step is None:
            self.finetune_start_step = checkpoint.get('global_step', 0)

    def on_save_checkpoint(self, checkpoint):
        """Save EMA teacher weights alongside the main state."""
        super().on_save_checkpoint(checkpoint)
        if self.target_ema_rate > 0:
            checkpoint['target_backbone_state_dict'] = self.target_backbone.state_dict()

    def validation_step(self, batch, batch_idx):
        """Log Δt observed during training when running validation."""
        output = super().validation_step(batch, batch_idx)
        if self.last_training_delta_t is not None:
            self.log('val/delta_t', self.last_training_delta_t, on_step=False, on_epoch=True, sync_dist=True)
        return output


class D3PMAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.noise.type == 'log-linear'
    assert self.parameterization == 'mean'

  def _process_model_output(self, model_output, xt, sigma):
    del xt 
    del sigma
    if self.subs_masking:
      model_output[:, :, self.mask_index] += self.neg_infinity
    return model_output.log_softmax(dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del dalpha_t
    assert not low_var
    dt = 1 / self.T
    t = 1 - alpha_t  # Only valid for log-linear schedule.
    t = t.clamp(0., 1.0 - 1e-4)
    alpha_t = alpha_t + torch.zeros_like(xt)
    alpha_s = t - dt + torch.zeros_like(xt)
    assert alpha_s.shape == xt.shape
    assert alpha_t.shape == xt.shape
    log_x_theta_at_x0 = torch.gather(
      log_x_theta, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(
      alpha_s * x_theta_at_m / (t - dt) + 1)
    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
    return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.predictor == 'analytic'

  def _get_score(self, x, sigma):
    return self.forward(x, sigma).exp()

  def _process_model_output(self, model_output, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(model_output.dtype)
    # logits shape
    # (batch_size, context_length, vocab_size)
    model_output = (model_output
                    - esigm1_log[:, None, None]
                    - np.log(model_output.shape[-1] - 1))
    # The below scatter operation sets the log score
    # for the input word to 0.
    model_output = torch.scatter(
      model_output, -1, xt[..., None],
      torch.zeros_like(model_output[..., :1]))
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    """Computes the SEDD loss for the Absorbing State Diffusion.

    Args:
      log_x_theta: float torch.Tensor with shape (batch_size,
          context_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          context_length), input.
      x0: int torch.Tensor with shape (batch_size,
          context_length), input.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
          time derivative of signal level.
      low_var: bool, low variance loss during training.
    
    Returns:
      loss with shape (batch_size, context_length).
    """
    assert not low_var
    masked_indices = xt == self.mask_index
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_x_theta[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_x_theta[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return dsigma * entropy


class DUO_BASE(trainer_base.UniformState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_save_checkpoint(checkpoint)

  def on_load_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
    super().on_load_checkpoint(checkpoint)

  def _process_model_output(self, model_output, xt, sigma):
    del xt, sigma
    return model_output.log_softmax(dim=-1)

  def _compute_posterior(self, x, xt, alpha_s, alpha_t):
    """Computes the posterior / approximate posterior.

    Args:
      x: Either clean input `x0` (one-hot),
        or model's predicted `x_theta` of shape (B, L, V).
      xt: The noisy latent (as indices) of shape (B, L).
      alpha_s: Noise level at s of shape (B, [L | 1], 1).
      alpha_t: Noise level at t of shape (B, [L | 1], 1).

    Returns:
      Posterior / approximate posterior of shape (B, L, V).
    """
    if self.config.sampling.use_float64:
      x = x.to(torch.float64)
    if alpha_s.ndim == 2:
      alpha_s = alpha_s.unsqueeze(-1)
    if alpha_t.ndim == 2:
      alpha_t = alpha_t.unsqueeze(-1)
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, self.vocab_size).to(
      self.dtype).to(self.device)
    return (
      (alpha_t * self.vocab_size * x * xt_one_hot + (
        alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
          1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
            alpha_t * self.vocab_size * torch.gather(
              x, -1, xt[..., None]) + (1 - alpha_t)))

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
    x_bar_theta = self.vocab_size * alpha_t[
        :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
    coeff = dalpha_t / (self.vocab_size * alpha_t)
    x_eq_xt = (x0 == xt).float()
    x_neq_xt = 1 - x_eq_xt
    xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
    xbar_theta_xt = torch.gather(
      x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
    xbar_theta_x = torch.gather(
      x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
    term1 = self.vocab_size * (1 / xbar_xt
                                - 1 / xbar_theta_xt)
    
    const = (1 - alpha_t) / (self.vocab_size * alpha_t
                             + 1 - alpha_t)
    term2_coefs = x_eq_xt * const + x_neq_xt
    term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                    - (1 / const) * x_neq_xt) * const.log()
    term2_theta = - term2_coefs * (
      x_bar_theta.log().sum(-1)
      - self.vocab_size * xbar_theta_xt.log())
    term2_theta = (
      term2_theta
      - self.vocab_size * alpha_t / (1 - alpha_t) * (
        xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
    term2 = term2_theta + term2_offset
    diffusion_loss = coeff * (term1 - term2)
    assert diffusion_loss.ndim == 2
    return diffusion_loss

  def _ancestral_update(self, x, t, dt, p_x0=None,
                   noise_removal_step=False):
    del p_x0
    _, alpha_t = self.noise(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t)
    assert alpha_t.ndim == 2
    
    q_xs = self._compute_posterior(
      x=self.forward(x, sigma_t).exp(),
      xt=x,
      alpha_s=alpha_s,
      alpha_t=alpha_t)
    if self.p_nucleus < 1:
      q_xs = utils.top_k_top_p_filtering(
        q_xs.log(), top_p=self.p_nucleus)
    return None, trainer_base.sample_categorical(q_xs, precision=self.config.sampling.gumbel_precision)


class Integral(torch.autograd.Function):
  """
  torch module calculating UDLM's p_t 
  """

  @staticmethod
  def forward(ctx, gamma_t, data):
    gamma_max = data['gamma_max']
    gamma_min = data['gamma_min']
    if (gamma_t.max() > gamma_max) or (
      gamma_t.min() < gamma_min):
      print('max:{} {}'.format(gamma_t.max(), gamma_max))
      print('min:{} {}'.format(gamma_t.min(), gamma_min))
      gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
    indices = torch.round(
      (data['num_points'] - 1) * (gamma_t - gamma_min) / (
          gamma_max - gamma_min)).long()
    grad_pt = data['grad_pt']
    ctx.grad_pt = grad_pt[indices]
    
    pt = data['pt'][indices]
    assert pt.shape == gamma_t.shape
    return pt

  @staticmethod
  def backward(ctx, grad_output):
    return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    with fsspec.open(self.config.algo.integral_cache_path,
                     'rb') as f:
      self.integral_cache = pickle.load(f)
    self.integral_cache['pt'] = torch.from_numpy(
      self.integral_cache['pt'])
    self.integral_cache['grad_pt'] = torch.from_numpy(
      self.integral_cache['grad_pt'])
    self.gamma_min = self.config.algo.gamma_min
    self.gamma_max = self.config.algo.gamma_max
    self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
    self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
    self.curriculum_start = self.config.algo.curriculum_start
    self.curriculum_end = self.config.algo.curriculum_end
    self.loss_type = self.config.algo.loss_type
    self._validate_configuration()

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.integral_cache['pt'] = self.integral_cache[
      'pt'].to(*args, **kwargs)
    self.integral_cache['grad_pt'] = self.integral_cache[
      'grad_pt'].to(*args, **kwargs)
    return self

  def _compute_gumbel_tau_inverse(self):
    start = self.gumbel_tau_log10_start
    end = self.gumbel_tau_log10_end
    delta = end - start
    if self.global_step < self.curriculum_start:
      tau = start
    elif self.global_step < self.curriculum_end:
      frac = (self.global_step - self.curriculum_start) / (
        self.curriculum_end - self.curriculum_start)
      tau = start + frac * delta
    else:
      tau = -10
    return 10 ** (-tau)

  def training_step(self, batch, batch_idx):
    self.log(name='gumbel_tau_log10',
             value=1 / self._compute_gumbel_tau_inverse(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return super().training_step(batch, batch_idx)

  def _gamma_to_alphat(self, gamma_t):
    integral = Integral.apply(gamma_t, self.integral_cache)
    return (self.vocab_size * integral - 1) / (
      self.vocab_size - 1)

  def _prior_loss(self):
    alpha_1 = self._gamma_to_alphat(
      torch.tensor(self.gamma_max))
    loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
      (self.vocab_size - 1) * alpha_1 + 1) + (
        1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1))
    return loss.item()

  def _q_xt_gaussian(self, x, gamma_t):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert x.ndim == 3
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()
    epsilon = torch.randn(x.shape, dtype=torch.float32,
                          device=self.device)
    return alpha_t * x + sigma_t * epsilon

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    use_true_nll = (self.global_step > self.curriculum_end
                    or not train_mode)
    if use_true_nll:
      return super().nll(x0, output_tokens,
                         current_accumulation_step)
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)    
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (
      self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(usdm_alpha_t)

    x0_one_hot = F.one_hot(x0, self.vocab_size)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
    xt = xt * self._compute_gumbel_tau_inverse()
    xt_usdm = xt.argmax(-1)
    log_x_theta = self.forward(xt, sigma=sigma)

    return self.nll_per_token(log_x_theta=log_x_theta,
                              xt=xt_usdm,
                              x0=x0,
                              alpha_t=usdm_alpha_t,
                              dalpha_t=usdm_dalpha_t,
                              low_var=False)


class Distillation(DUO):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.update_teacher_every = config.algo.update_teacher_every
    self.save_hyperparameters()
    self.teacher = None
    self.teacher_ema = config.algo.teacher_ema
    self.linear_growth_dt = config.algo.linear_growth_dt
    self.linear_growth_min = config.algo.linear_growth_min
    self.linear_growth_max = config.algo.linear_growth_max

  def _validate_configuration(self):
    assert os.path.exists(
      self.config.algo.integral_cache_path), (
        'The integral cache (Eq. 10 in the paper) for '
        f'the {self.config.data.tokenizer_name_or_path} '
        ' tokenizer doesnt exist at '
        f'{self.config.algo.integral_cache_path}. '
        'Please generate it by running the utils.py script, '
        'and ensure the correct path is specified using the '
        'algo.integral_cache_path flag.')
    assert self.loss_type in {
      'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

  def _maybe_update_teacher_weights(self):
    if self.global_step % self.update_teacher_every != 0:
      return
    if self.teacher_ema:
      self.ema.copy_to(self.teacher.parameters())
    else:
      for better_param, current_param in zip(
        self.backbone.parameters(), self.teacher.parameters()):
        if current_param.requires_grad:
          current_param.data.copy_(better_param.data)

  @torch.no_grad()
  def _teacher_logits(self, xt, sigma):
    if self.teacher is None:
      self.teacher = copy.deepcopy(self.backbone)
    self._maybe_update_teacher_weights()

    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      model_output = self.teacher(xt, sigma)
    logits = self._process_model_output(
      model_output=model_output, xt=xt, sigma=sigma)
    return logits.detach()

  def _sample_trajectory(self, x0, gamma_t, gamma_s):
    """Computes the noisy sample xt."""
    assert gamma_t.ndim == 1
    assert gamma_s.ndim == 1
    assert x0.ndim == 2
    x0 = F.one_hot(x0, self.vocab_size).to(
      self.dtype).to(self.device)
    gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
    alpha_t = torch.sigmoid(-gamma_t).sqrt()
    sigma_t = torch.sigmoid(gamma_t).sqrt()

    gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
    alpha_s = torch.sigmoid(-gamma_s).sqrt()
    sigma_s = torch.sigmoid(gamma_s).sqrt()
    
    epsilon = torch.randn(x0.shape, dtype=torch.float32,
                          device=self.device)
    xt = alpha_t * x0 + sigma_t * epsilon
    xs = alpha_s * x0 + sigma_s * epsilon
    return xt, xs

  def _compute_dt(self):
    if self.linear_growth_dt:
      scale = self.global_step / self.trainer.max_steps
      return self.linear_growth_min + scale * (
        self.linear_growth_max -  self.linear_growth_min)
    n = self.global_step // self.update_teacher_every
    return 2 ** n / self.T

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=None):
    del output_tokens, train_mode
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dt = self._compute_dt()
    t = torch.clip(t + dt, 0, 1)

    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)
    gamma_s = self.gamma_min + (
      t - dt) * (self.gamma_max - self.gamma_min)

    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    usdm_alpha_s = self._gamma_to_alphat(gamma_s)
    usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
    assert usdm_alpha_s.ndim == 2

    xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
    xt_discrete = xt.argmax(-1)
    xs_discrete = xs.argmax(-1)
    log_x_theta_student = self.forward(
      xt_discrete, sigma=self._sigma_from_alphat(usdm_alpha_t))
    log_x_theta_teacher = self._teacher_logits(
      xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
    if self.config.training.loss_precision == 'float64':
      log_x_theta_student = log_x_theta_student.to(torch.float64)
      log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
    if self.loss_type == 'kl-fwd':
      return (log_x_theta_teacher.exp() * (
        log_x_theta_teacher - log_x_theta_student)).sum(-1)
    elif self.loss_type == 'kl-bwd':
      return (log_x_theta_student.exp() * (
        log_x_theta_student - log_x_theta_teacher)).sum(-1)
    
  def training_step(self, batch, batch_idx):
    self.log(name='dt',
             value=self._compute_dt(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return super().training_step(batch, batch_idx)
