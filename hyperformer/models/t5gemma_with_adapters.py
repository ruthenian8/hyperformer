import logging
from typing import Any, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def is_t5gemma_config(config) -> bool:
    return getattr(config, "model_type", None) in {"t5gemma", "t5_gemma", "t5-gemma"}


def get_model_hidden_size(config) -> int:
    if hasattr(config, "d_model") and config.d_model is not None:
        return int(config.d_model)

    if hasattr(config, "hidden_size") and config.hidden_size is not None:
        return int(config.hidden_size)

    for side in ("encoder", "decoder"):
        side_cfg = getattr(config, side, None)
        if side_cfg is not None:
            if hasattr(side_cfg, "hidden_size") and side_cfg.hidden_size is not None:
                return int(side_cfg.hidden_size)
            if hasattr(side_cfg, "d_model") and side_cfg.d_model is not None:
                return int(side_cfg.d_model)

    raise ValueError(
        f"Could not infer hidden size from config of type {type(config)}. "
        f"Available attrs: {sorted(list(vars(config).keys()))[:50]}"
    )


def _iter_blocks(module: nn.Module):
    """
    Best-effort block discovery for T5Gemma.
    We look for modules that contain BOTH an attention submodule and an MLP/FFN submodule.
    This avoids hardcoding internal class names (which vary by Transformers version).
    """
    for name, m in module.named_modules():
        has_attn = any(hasattr(m, k) for k in ("self_attn", "attn", "attention"))
        has_mlp = any(hasattr(m, k) for k in ("mlp", "ffn", "feed_forward"))
        if has_attn and has_mlp:
            yield name, m


class T5GemmaForConditionalGenerationWithAdapters(nn.Module):
    """
    Thin wrapper around HF's T5GemmaForConditionalGeneration that:
    - accepts `task=` like the repo's modified T5 model
    - applies HyperFormer AdapterControllers via forward hooks

    NOTE: This expects you are running a Transformers version that provides:
      `transformers.T5GemmaForConditionalGeneration`
    """
    def __init__(self, config, adapter_config=None):
        super().__init__()
        from transformers import T5GemmaForConditionalGeneration

        self.model = T5GemmaForConditionalGeneration(config)
        self.config = self.model.config

        self._current_task: Optional[str] = None
        self._hooks = []

        self.adapter_config = adapter_config
        if adapter_config is not None:
            self._setup_adapters()

    def _clear_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _setup_adapters(self):
        from hyperformer.adapters import AdapterController

        self._clear_hooks()
        hidden = get_model_hidden_size(self.config)
        self.adapter_config.input_dim = hidden

        self.attn_adapters = nn.ModuleList()
        self.ffn_adapters = nn.ModuleList()

        blocks = list(_iter_blocks(self.model))
        if not blocks:
            raise RuntimeError(
                "Could not find transformer blocks in T5Gemma model. "
                "You may need to update _iter_blocks() for your Transformers version."
            )

        for _, _ in blocks:
            self.attn_adapters.append(AdapterController(self.adapter_config))
            self.ffn_adapters.append(AdapterController(self.adapter_config))

        for idx, (_, block) in enumerate(blocks):
            attn_mod = None
            for k in ("self_attn", "attn", "attention"):
                if hasattr(block, k):
                    attn_mod = getattr(block, k)
                    break

            mlp_mod = None
            for k in ("mlp", "ffn", "feed_forward"):
                if hasattr(block, k):
                    mlp_mod = getattr(block, k)
                    break

            if attn_mod is None or mlp_mod is None:
                continue

            self._hooks.append(attn_mod.register_forward_hook(self._make_attn_hook(idx)))
            self._hooks.append(mlp_mod.register_forward_hook(self._make_ffn_hook(idx)))

        logger.info("Attached adapters to %d discovered blocks.", len(blocks))

    @classmethod
    def from_pretrained(cls, model_path: str, config, cache_dir=None, adapter_config=None, **kwargs):
        from transformers import T5GemmaForConditionalGeneration

        obj = cls(config=config, adapter_config=adapter_config)
        obj.model = T5GemmaForConditionalGeneration.from_pretrained(
            model_path, config=config, cache_dir=cache_dir, **kwargs
        )
        obj.config = obj.model.config
        if adapter_config is not None:
            obj._setup_adapters()
        return obj

    def _make_attn_hook(self, idx: int):
        def hook(_module, _inputs, output):
            if self.adapter_config is None:
                return output
            if self._current_task is None:
                return output

            if isinstance(output, tuple):
                h = output[0]
                h2 = self.attn_adapters[idx](self._current_task, h)
                return (h2,) + output[1:]
            return self.attn_adapters[idx](self._current_task, output)
        return hook

    def _make_ffn_hook(self, idx: int):
        def hook(_module, _inputs, output):
            if self.adapter_config is None:
                return output
            if self._current_task is None:
                return output

            if isinstance(output, tuple):
                h = output[0]
                h2 = self.ffn_adapters[idx](self._current_task, h)
                return (h2,) + output[1:]
            return self.ffn_adapters[idx](self._current_task, output)
        return hook

    def forward(self, *args, task: Optional[str] = None, task_embedding: Any = None, **kwargs):
        self._current_task = task
        out = self.model(*args, **kwargs)
        self._current_task = None
        return out

    def generate(self, *args, task: Optional[str] = None, **kwargs):
        self._current_task = task
        out = self.model.generate(*args, **kwargs)
        self._current_task = None
        return out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
