# -*- coding: utf-8 -*-
# src/model.py — DANN Text Detector với LoRA + Gradient Reversal Layer (Dynamic Model Support)

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANN_TextDetector(nn.Module):
    def __init__(self, model_name: str, num_classes: int, num_domains: int,
                 dropout: float = 0.3, use_lora: bool = True, use_dann: bool = True):
        super().__init__()
        self.use_dann    = use_dann
        self.transformer = AutoModel.from_pretrained(model_name)

        if use_lora:
            # DYNAMIC MAPPING: Tự động nhận diện target_modules theo kiến trúc backbone
            model_name_lower = model_name.lower()
            if "deberta" in model_name_lower:
                target_modules = ["query_proj", "value_proj"]
            elif "roberta" in model_name_lower:
                target_modules = ["query", "value"]
            else:
                target_modules = ["query", "value"]  # Cấu hình fallback mặc định

            peft_config = LoraConfig(
                task_type      = TaskType.FEATURE_EXTRACTION,
                r              = 8,
                lora_alpha     = 16,
                target_modules = target_modules,
                lora_dropout   = 0.1,
                modules_to_save= ["class_head", "domain_head"],
            )
            self.transformer = get_peft_model(self.transformer, peft_config)

        # DYNAMIC HIDDEN SIZE: Tự động bốc kích thước ẩn từ config gốc (768, 1024, v.v.)
        hidden = self.transformer.config.hidden_size

        # Nhánh phân loại chính (Human vs Machine)
        self.class_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        # Domain head + GRL (chỉ active khi use_dann=True)
        self.grl = GradientReversalLayer(alpha=0.0)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def update_alpha(self, new_alpha: float):
        self.grl.alpha = new_alpha

    def forward(self, input_ids, attention_mask):
        outputs    = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        class_logits  = self.class_head(cls_output)
        domain_logits = self.domain_head(self.grl(cls_output)) if self.use_dann else None
        return class_logits, domain_logits

def compute_lambda(epoch: int, batch_idx: int, num_batches: int,
                   total_epochs: int, gamma: float = 10.0) -> float:
    """Lambda schedule từ paper gốc DANN — tăng dần từ 0 đến 1."""
    p = (epoch * num_batches + batch_idx) / (total_epochs * num_batches)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
