import os
import json
import torch


class Config:
    def __init__(self):
        self.input = None
        self.output = None
        self.vocab_size = 32000
        self.num_workers = 4
        self.entries = 8000

        self.batch_size = 4
        self.context_length = 256
        self.embedding_dimension = 128
        self.num_heads = 1
        self.num_layers = 1
        self.dropout = 0.2

        self.learning_rate = 0.0001
        self.weight_decay = 0.01
        self.T_0 = 3
        self.T_mult = 2
        self.eta_min = 0.00001
        self.lr_decay = 0.75

        self.max_iterations = 1000
        self.log_interval = 10
        self.eval_interval = 10
        self.precision = 16

        self.gradient_clip_val = 1
        self.devices = 1

        self.deepspeed = {
            "zero_allow_untested_optimizer": True,
            "prescale_gradients": False,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": False,
                "allgather_bucket_size": 5e8,
                "reduce_bucket_size": 5e8,
                "overlap_comm": True,
            },
        }

    def get_dtype(self, precision):
        if precision == "64" or precision == 64 or precision == "64-true":
            self.deepspeed = None
            return torch.float64
        elif precision == "32" or precision == 32 or precision == "32-true":
            return torch.float32
        elif (
            precision == "16"
            or precision == 16
            or precision == "16-true"
            or precision == "16-mixed"
        ):
            return torch.float16
        elif (
            precision == "bf16-true" or precision == "bf16-mixed" or precision == "bf16"
        ):
            self.deepspeed["bf16"] = {"enabled": True}
            return torch.bfloat16
        elif precision == "transformer-engine":
            self.deepspeed = None
            return torch.bfloat16
        elif precision == "transformer-engine-float16":
            self.deepspeed = None
            return torch.float16
        else:
            raise ValueError(f"Precision {precision} not supported.")

    def verify(self):
        if self.input is None:
            raise ValueError("Input not set.")
        if self.output is None:
            raise ValueError("Output not set.")
        if not (os.path.exists(self.output) and os.path.isdir(self.output)):
            os.mkdir(self.output)

        self.dtype = self.get_dtype(self.precision)
        self.train_bin_path = self.output + "/train"
        self.val_bin_path = self.output + "/val"
        self.test_bin_path = self.output + "/test"


class ConfigReader:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, "r") as f:
            config = json.load(f)
            self.config = config
        self.result = Config()

    def read(self):
        for key, value in self.config.items():
            setattr(self.result, key, value)
        self.result.verify()
        return self.result
