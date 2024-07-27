import os
from typing import Optional, Dict, Union
from huggingface_hub import hf_hub_download

import torch
from safetensors import safe_open
from diffusers import UNet2DConditionModel
from diffusers.loaders.lora import LORA_WEIGHT_NAME_SAFE

from torch import nn

class ZipLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_merger_value: Optional[float] = 1.0,
        init_merger_value_2: Optional[float] = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight_1 = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.weight_2 = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.merger_1 = nn.Parameter(
            torch.ones((in_features,), device=device, dtype=dtype) * init_merger_value
        )
        self.merger_2 = nn.Parameter(
            torch.ones((in_features,), device=device, dtype=dtype) * init_merger_value_2
        )
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        assert type in ["merge", "weight_1", "weight_2"]
        self.forward_type = type

    def compute_mergers_similarity(self):
        return (self.merger_1 * self.merger_2).abs().mean()

    def get_ziplora_weight(self):
        return self.merger_1 * self.weight_1 + self.merger_2 * self.weight_2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1.dtype
        if self.forward_type == "merge":
            weight = self.get_ziplora_weight()
        elif self.forward_type == "weight_1":
            weight = self.weight_1
        elif self.forward_type == "weight_2":
            weight = self.weight_2
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class ZipLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)


def get_lora_weights(
    lora_name_or_path: str, subfolder: Optional[str] = None, **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Args:
        lora_name_or_path (str): huggingface repo id or folder path of lora weights
        subfolder (Optional[str], optional): sub folder. Defaults to None.
    """
    if os.path.exists(lora_name_or_path):
        if subfolder is not None:
            lora_name_or_path = os.path.join(lora_name_or_path, subfolder)
        if os.path.isdir(lora_name_or_path):
            lora_name_or_path = os.path.join(lora_name_or_path, LORA_WEIGHT_NAME_SAFE)
    else:
        lora_name_or_path = hf_hub_download(
            repo_id=lora_name_or_path,
            filename=LORA_WEIGHT_NAME_SAFE,
            subfolder=subfolder,
            **kwargs,
        )
    assert lora_name_or_path.endswith(
        ".safetensors"
    ), "Currently only safetensors is supported"
    tensors = {}
    with safe_open(lora_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def merge_lora_weights(
    tensors: torch.Tensor, key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        merged_weight = tensors[up_key] @ tensors[down_key]
        out[part] = merged_weight
    return out


def initialize_ziplora_layer(state_dict, state_dict_2, part, **model_kwargs):
    ziplora_layer = ZipLoRALinearLayer(**model_kwargs)
    ziplora_layer.load_state_dict(
        {
            "weight_1": state_dict[part],
            "weight_2": state_dict_2[part],
        },
        strict=False,
    )
    return ziplora_layer


def unet_ziplora_state_dict(
    unet: UNet2DConditionModel, quick_release: bool = False
) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "get_ziplora_weight"), lora_layer
                weight = lora_layer.get_ziplora_weight()
                lora_state_dict[f"unet.{name}.lora.weight"] = weight

                if quick_release:
                    lora_layer.cpu()
    return lora_state_dict


def ziplora_set_forward_type(unet: UNet2DConditionModel, type: str = "merge"):
    assert type in ["merge", "weight_1", "weight_2"]

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "set_forward_type"), lora_layer
                lora_layer.set_forward_type(type)
    return unet


def ziplora_compute_mergers_similarity(unet):
    similarities = []
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "compute_mergers_similarity"), lora_layer
                similarities.append(lora_layer.compute_mergers_similarity())
    similarity = torch.stack(similarities).sum(dim=0)
    return similarity


def merge_lora_weights_for_inference(
    tensors: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        key = target_key + f".{part}.lora.weight"
        out[part] = tensors[key]
    return out


def initialize_ziplora_layer_for_inference(state_dict, part, **model_kwargs):
    ziplora_layer = ZipLoRALinearLayerInference(**model_kwargs)
    ziplora_layer.load_state_dict(
        {
            "weight": state_dict[part],
        },
        strict=False,
    )
    return ziplora_layer


def insert_ziplora_to_unet(
    unet: UNet2DConditionModel, ziplora_name_or_path: str, **kwargs
):
    tensors = get_lora_weights(ziplora_name_or_path, **kwargs)
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        state_dict = merge_lora_weights_for_inference(tensors, key=attn_name)
        # Set the `lora_layer` attribute of the attention-related matrices.
        kwargs = {"state_dict": state_dict}

        attn_module.to_q.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                part="to_q",
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                part="to_k",
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                part="to_v",
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                part="to_out.0",
                **kwargs,
            )
        )
    return unet