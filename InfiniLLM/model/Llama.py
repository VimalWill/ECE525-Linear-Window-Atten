import json
import torch
from torch import nn
from pathlib import Path

from .ops import Transformer, ModelArgs

class LLama:
    @staticmethod
    def build(chkpt_dir: str, 
              model_seq_len: int, 
              model_batch_size: int, 
              use_cache:bool = False,
              attn_method:str = "softmax", 
              window_size:int = 128,
              delta_values:int = None):
        
        checkpoints = sorted(Path(chkpt_dir).glob("*.pth"))
        ckpt_path = checkpoints[0] # no parallel opt
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # setting up the model configuration
        with open(Path(chkpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args = ModelArgs(
            max_seq_len=model_seq_len, 
            max_batch_size=model_batch_size, 
            **params, 
        )

        if use_cache:
            print("using cache!")

        model = Transformer(model_args, use_cache, attn_method, window_size, delta_values)
        model.load_state_dict(checkpoint, strict=False)

        return model