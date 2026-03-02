import abc
from typing import Optional
import torch
from memory_policy import MemoryPolicy


class MemoryModelWrapper(abc.ABC):
    def __init__(self,
                 config,
                 memory_policy: MemoryPolicy, 
                 registration_kwargs: dict,
                 # if set, recomputes memory policy every fixed number of steps
                 memory_policy_fixed_delay: Optional[int] = None,
                 output_attentions_full_precision: bool = True,
                 max_new_tokens: Optional[int] = None,
                 ):
        self.memory_policy: MemoryPolicy = memory_policy
        self.registration_kwargs = registration_kwargs
        self.config = config
        self.memory_policy.register_new_memory_model(
            self.config, self.registration_kwargs)
        self.memory_policy.finalize_registration()
        self.memory_policy_fixed_delay = memory_policy_fixed_delay
        
        if not hasattr(self, 'max_new_tokens'):
            self.max_new_tokens = max_new_tokens
        
        if (self.memory_policy_fixed_delay is not None) and (
            self.max_new_tokens is not None):
            assert self.memory_policy_fixed_delay % self.max_new_tokens == 0
        self.output_attentions_full_precision = output_attentions_full_precision
        self._past_length = 0
    
    def load_partial_state_dict(self, state_dict):
        current_state = self.state_dict()
        for name in self.base_model_param_keys:
            target_param = state_dict[name]
            current_state[name].copy_(target_param.data)


    def swap_memory_policy(self, new_memory_policy: MemoryPolicy):
        self.memory_policy = new_memory_policy
        self.memory_policy.register_new_memory_model(
            self.config, self.registration_kwargs)
        self.memory_policy.finalize_registration()
        self.memory_requires_attn = self.memory_policy.requires_attn_scores
        self.memory_requires_queries = self.memory_policy.requires_queries
    
    @property
    def cache_size(self,):
        return self.memory_policy.cache_size
    
    def set_memory_params(self, params) -> None:
        self.memory_policy.set_params(params=params)

    def get_memory_params(self,):
        return self.memory_policy.get_layer_params()

    def set_memory_params_batch_idxs(self, param_idxs) -> None:
        self.memory_policy.set_params_batch_idxs(param_idxs=param_idxs)
    
    def get_param_size(self,):
        return self.memory_policy.param_size

    def get_param_stats(self,):
        return self.memory_policy.get_param_stats()
    
    def get_buffers_list(self,):
        return self.memory_policy.get_buffers_list()

    def self_merge(self,):
        return self.memory_policy.self_merge()
    
    def merge_buffers_list(self, buffers_to_merge):
        return self.memory_policy.merge_buffers_list(
            buffers_to_merge=buffers_to_merge)

    def receive_buffers_list(self, buffers_list):
        return self.memory_policy.receive_buffers_list(
            buffers_list=buffers_list)
    
    # functions to save (e.g., normalization) buffers along with the checkpoints
    def get_buffers_dict(self,):
        return self.memory_policy.get_buffers_dict()
    
    def load_buffers_dict(self, buffers_dict):
        return self.memory_policy.load_buffers_dict(buffers_dict=buffers_dict)
    
    def memory_policy_has_buffers_to_merge(self,):
        return self.memory_policy._has_buffers_to_merge
    
    # functions to signal the memory policy that ut is being trained/evaluated
    def training_mode(self,):
        self.memory_policy.training_mode()

    def evaluation_mode(self,):
        self.memory_policy.evaluation_mode()

    def freeze_sync_buffers(self, freeze=True):
        self.memory_policy.freeze_sync_buffers(freeze=freeze)

    def unfreeze_sync_buffers(self,):
        self.memory_policy.unfreeze_sync_buffers()
#
    def are_sync_buffers_frozen(self,):
        return self.memory_policy.are_sync_buffers_frozen()

    def _get_lora_params(self):
        """Enumerate LoRA-only parameters (requires_grad=True).

        After apply_lora_adapters(), PEFT sets requires_grad=False on all base
        model parameters and requires_grad=True only on LoRA A/B matrices.
        This makes requires_grad the authoritative LoRA discriminator.

        Returns:
            List of parameter tensors (in model.parameters() iteration order).
            Order is deterministic for a fixed PEFT version — do NOT sort by name.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_lora_params_flat(self):
        """Return LoRA weights as a new detached CPU float32 flat tensor.

        Returns a copy (not a view). Safe to modify without affecting the model.
        Returns an empty tensor if no LoRA adapters have been injected.
        """
        lora_params = self._get_lora_params()
        if not lora_params:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat([
            p.detach().cpu().float().flatten() for p in lora_params
        ])

    def set_lora_params(self, flat_vec):
        """Restore LoRA weights from a flat float32 vector.

        Casts flat_vec values back to each parameter's dtype before writing,
        preserving the float32 storage for LoRA while correctly handling
        devices (writes to wherever the parameter currently lives).

        Args:
            flat_vec: 1D tensor of length == total LoRA parameter count.

        Raises:
            ValueError: If flat_vec length does not match total LoRA params.
        """
        lora_params = self._get_lora_params()
        total = sum(p.numel() for p in lora_params)
        if len(flat_vec) != total:
            raise ValueError(
                f"set_lora_params: flat_vec size {len(flat_vec)} != "
                f"total_lora_params {total}"
            )
        offset = 0
        for p in lora_params:
            numel = p.numel()
            p.data.copy_(
                flat_vec[offset:offset + numel]
                .reshape(p.shape)
                .to(dtype=p.dtype, device=p.device)
            )
            offset += numel

class MemoryDecoderLayer(abc.ABC):
    
    abc.abstractmethod
    def __init__(self,):
        pass

class MemoryAttention(abc.ABC):
    abc.abstractmethod
    def __init__(self,):
        pass
   