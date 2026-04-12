import os
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict


import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from .base import MemoryPolicy, ParamMemoryPolicy
from .base_dynamic import (
    DynamicMemoryPolicy, DynamicParamMemoryPolicy,
    RecencyParams, AttentionParams, threshold_score_idxs, is_tpu,
    )
from  .base_deep_components import SelectionNetwork, ComponentOutputParams


def _apply_cache_validity_mask(masked_full_scores, cache_validity_mask):
    """On TPU, mask out previously-evicted cache entries so they score min."""
    if cache_validity_mask is None:
        return masked_full_scores
    min_value = torch.finfo(masked_full_scores.dtype).min
    n_kv = masked_full_scores.shape[-1]
    n_old = cache_validity_mask.shape[-1]
    # Pad: old cache validity + True for all new tokens
    if n_old < n_kv:
        full_validity = F.pad(
            cache_validity_mask, (0, n_kv - n_old), value=True)
    else:
        full_validity = cache_validity_mask
    return torch.where(full_validity, masked_full_scores, min_value)


class DynamicSelection(SelectionNetwork):
    '''Replicates the default selection criteria of dynamic memory policies'''
    def __init__(self,
                 per_layer: bool,
                 per_head: bool,
                 shared: bool,
                 cache_size: Optional[int],
                 dynamic_thresh: float, 
                 ):
        SelectionNetwork.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=ComponentOutputParams(
                requires_recomputation=False,
                reduction_mode=None,
                ema_params=None,
                output_past_non_reduced_history=False,
                max_non_reduced_history_len=None,
            ),
            buffer_names=[],
            initializer=dynamic_thresh,
            )
        self.cache_size = cache_size

    def select_new_tokens(
        self,
        parameters,
        token_scores,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift: float = 0.0,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces indexes for the selected KV cache tokens and a selection
           mask.'''
        dynamic_thresh = parameters
        min_value = torch.finfo(token_scores.dtype).min
        max_value = torch.finfo(dynamic_thresh.dtype).max
        
        
        masked_full_scores = torch.where(
            attn_mask.bool(), token_scores, min_value)
        masked_full_scores = _apply_cache_validity_mask(
            masked_full_scores, kwargs.get('cache_validity_mask'))

        masked_full_scores[..., -1] = max_value
        retained_idxs, new_mask = threshold_score_idxs(
            masked_full_scores=masked_full_scores,
            dynamic_thresh=dynamic_thresh + threshold_shift,
            preserve_order=True,
            cache_size=self.cache_size,
            )
        
        
        return retained_idxs, new_mask
    
    def get_cache_size(self,) -> Optional[int]:
        return self.cache_size

    def net_param_size(self,) -> int:
        return 1

    def get_param_scaling(self,) -> Optional[Union[str, Tuple[float, float]]]:
        return 'exp'


class FixedThresholdSelection(DynamicSelection):
    """DynamicSelection with the threshold frozen at its initial value.

    The threshold is NOT included in the CMA-ES parameter vector (net_param_size=0),
    so CMA-ES only evolves the scoring network. The fixed threshold is stored as
    self._fixed_thresh (set from the config's dynamic_thresh, default 0).

    Use this to match the original NAMM paper's setup where the threshold is
    static at 0 and only the scoring head is learned.
    """

    def net_param_size(self) -> int:
        return 0  # threshold is not a CMA-ES parameter

    def select_new_tokens(self, parameters, token_scores, **kwargs):
        # `parameters` is empty (size 0) because net_param_size=0.
        # Build a scalar threshold tensor from the config init value.
        fixed = torch.tensor(
            self.initializer, dtype=token_scores.dtype,
            device=token_scores.device)
        return super().select_new_tokens(
            parameters=fixed, token_scores=token_scores, **kwargs)
        
        
        return 'exp'

class BinarySelection(SelectionNetwork):
    '''Mantains tokens when scores > 0 - can be probabilistic, based on a 
       logistic distribution instead'''
    
    def __init__(self,
                 per_layer: bool,
                 per_head: bool,
                 shared: bool,
                 cache_size: Optional[int],
                 is_probabilistic: bool = False,
                 temp: float = 1.0, 
                 learned_temp: bool = False, 
                 ):
        
        if learned_temp:
            assert is_probabilistic, (
                'If is not probabilistic, the temperature will not be used')
            self._needs_learned_temp = True
        else:
            self._needs_learned_temp = False
        self.is_probabilistic = is_probabilistic
        self.initial_temp = temp
        SelectionNetwork.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=ComponentOutputParams(
                requires_recomputation=False,
                reduction_mode=None,
                ema_params=None,
                output_past_non_reduced_history=False,
                max_non_reduced_history_len=None,
            ),
            buffer_names=[],
            initializer=temp,
            )
        self.cache_size = cache_size
        

    def select_new_tokens(
        self,
        parameters,
        token_scores,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift: float = 0.0,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces indexes for the selected KV cache tokens and a selection
           mask.'''
        
        
        
        
        
        min_value = torch.finfo(token_scores.dtype).min
        max_value = torch.finfo(token_scores.dtype).max

        if self.is_probabilistic:
            if self._needs_learned_temp:
                temp = parameters
            else:
                temp = self.initial_temp
            masked_full_scores = torch.where(
                attn_mask.bool(), token_scores, min_value)
            probabilities = F.sigmoid(masked_full_scores / temp)
            random_samples = torch.rand_like(probabilities)
            token_scores = (probabilities >= random_samples).to(
                probabilities.dtype) - 0.5

        masked_full_scores = torch.where(
            attn_mask.bool(), token_scores, min_value)
        masked_full_scores = _apply_cache_validity_mask(
            masked_full_scores, kwargs.get('cache_validity_mask'))
        masked_full_scores[..., -1] = max_value
        retained_idxs, new_mask = threshold_score_idxs(
            masked_full_scores=masked_full_scores,
            dynamic_thresh=threshold_shift,
            preserve_order=True,
            cache_size=self.cache_size,
            )
        
        
        return retained_idxs, new_mask
    
    def get_cache_size(self,) -> Optional[int]:
        return self.cache_size
    
    def net_param_size(self,) -> int:
        if self._needs_learned_temp:
            return 1
        else:
            return 0
    
    def get_param_scaling(self,) -> Optional[Union[str, Tuple[float, float]]]:
        
        
        return 'exp'


class TopKSelection(SelectionNetwork):
    '''Simply collects the top K scores with no thesholding'''
    def __init__(self,



                 cache_size: Optional[int],
                 protected_tail_n: int = 0,

                 ):
        SelectionNetwork.__init__(
            self,
            per_layer=False,
            per_head=False,
            shared=True,
            output_params=ComponentOutputParams(
                requires_recomputation=False,
                reduction_mode=None,
                ema_params=None,
                output_past_non_reduced_history=False,
                max_non_reduced_history_len=None,
            ),
            buffer_names=[],
            )
        self.cache_size = cache_size
        self.protected_tail_n = protected_tail_n

    def select_new_tokens(
        self,
        parameters,
        token_scores,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces indexes for the selected KV cache tokens and a selection
           mask.'''

        num_samples = token_scores.shape[-1]
        if self.cache_size is not None and num_samples > self.cache_size:
            min_value = torch.finfo(token_scores.dtype).min
            masked_full_scores = torch.where(
                attn_mask.bool(), token_scores, min_value)
            masked_full_scores = _apply_cache_validity_mask(
                masked_full_scores, kwargs.get('cache_validity_mask'))
            # Protect the last N positions (e.g. chat template tail) from
            # eviction by setting their scores to +inf. This guarantees
            # they survive topk selection without changing the learned
            # scoring policy or requiring retraining.
            if self.protected_tail_n > 0:
                max_value = torch.finfo(token_scores.dtype).max
                tail_start = max(0, num_samples - self.protected_tail_n)
                masked_full_scores[..., tail_start:] = max_value
            _, retained_idxs = torch.topk(
                masked_full_scores, k=self.cache_size, sorted=False, dim=-1)
            retained_idxs, _ = retained_idxs.sort(descending=False, dim=-1,)
        else:
            retained_idxs = torch.arange(
                num_samples, device=token_scores.device,).view(
                    1, 1, num_samples).expand_as(token_scores)
        if self.cache_size is not None:
            attn_mask = attn_mask[..., -self.cache_size:]
        new_mask = torch.ones_like(retained_idxs, dtype=torch.bool)*attn_mask
        return retained_idxs, new_mask
    
    def get_cache_size(self,) -> Optional[int]:
        return self.cache_size
    
    def net_param_size(self,) -> int:
        return 0
    
    
    
    
    
