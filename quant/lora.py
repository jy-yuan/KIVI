import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math

from qmodule import WQLinearForTrain

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False  # BA whether merged with W_0
        self.merge_weights = merge_weights

class QLoRALinear(nn.Module, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        qlinear: WQLinearForTrain,
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.qlinear = qlinear
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features, dtype=torch.float))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.qlinear.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

        # seems not work
        assert fan_in_fan_out == False, "fan_in_fan_out is not supported yet"
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        # nn.Linear.train(self, mode)
        # Merge not implemented
        # if mode:
        #     if self.merge_weights and self.merged:
        #         # Make sure that the weights are not merged
        #         if self.r > 0:
        #             self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = False
        # else:
        #     if self.merge_weights and not self.merged:
        #         # Merge the weights and mark it
        #         if self.r > 0:
        #             self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = True       

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = self.qlinear(x)
            x = self.lora_dropout(x)
            result += (x.float() @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return self.qlinear(x)