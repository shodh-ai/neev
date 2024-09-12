from .data import DataModule
from .model import Transformer
from .utils.misc import measure_time
from .data.tokenizer import Tokenizer,JsonlTokenizer

from .utils.args_parser import training_parser


import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class DataPreparer:
    def __init__(self, input_dir=None, output_dir=None, config_path=None, parser=None):
        parser = self.check_args(input_dir, output_dir, config_path, parser)
        config = training_parser(parser)
        self.config = config
        self.input_dir = config.input
        self.output_dir = config.output
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.context_length  = config.context_length
        self.vocab_save_path = config.vocab_save_path
        self.tokenizer_type = config.tokenizer_type
        self.tokenizer = JsonlTokenizer(self.input_dir,self.output_dir ,self.vocab_save_path,self.tokenizer_type,self.vocab_size)


        
    def setup(self):
        self.tokenizer.tokenize()
    def check_args(self, input_dir, output_dir, config_path, parser):
        if parser is None:
            parser = ArgumentParser()
            if input_dir is not None:
                parser.add_argument("--input", type=str, default=input_dir)
            if output_dir is not None:
                parser.add_argument("--output", type=str, default=output_dir)
            if config_path is not None:
                parser.add_argument("--config", type=str, default=config_path)
        return parser

class NeevTrainer:
    def __init__(self, input_dir=None, output_dir=None, config_path=None, parser=None):

        parser = self.check_args(input_dir, output_dir, config_path, parser)
        torch.set_float32_matmul_precision("high")
        self.lr_monitor = LearningRateMonitor(logging_interval="step")
        config = training_parser(parser)
        self.config = config

        self.input_dir = config.input
        self.output_dir = config.output
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.context_length  = config.context_length
        self.vocab_save_path = config.vocab_save_path
        self.tokenizer_type = config.tokenizer_type
        self.devices = config.devices
        self.gradient_clip_val = config.gradient_clip_val
        
        self.start_time = measure_time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = TensorBoardLogger("logs/", name="transformer")
        self.version = self.logger.version
        self.profiler_log_dir = f"logs/profiler/version_{self.version}"
        self.profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.profiler_log_dir
            ),
            trace_memory=True,
            export_to_chrome=True,
        )

        self.dataModule = DataModule(self.output_dir,self.batch_size, self.vocab_size,self.context_length)
        if config.deepspeed is not None:
            strategy = DeepSpeedStrategy(config=config.deepspeed)
        else:
            strategy = "ddp"
        

        self.checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"logs/checkpoints/",
        filename="checkpoint-step-{step:08d}",
        save_top_k=-1,
        mode="min",
    )

        self.trainer = Trainer(
            accelerator="auto",
            devices=config.devices,
            max_epochs=config.max_epochs,
            min_epochs=config.min_epochs,
            max_steps=config.max_iterations,
            precision=config.precision,
            log_every_n_steps=config.log_interval,
            strategy=strategy,
            logger=self.logger,
            profiler=self.profiler,
            callbacks=[self.lr_monitor, self.checkpoint],
            gradient_clip_val=config.gradient_clip_val,
        )

    def check_args(self, input_dir, output_dir, config_path, parser):
        if parser is None:
            parser = ArgumentParser()
            if input_dir is not None:
                parser.add_argument("--input", type=str, default=input_dir)
            if output_dir is not None:
                parser.add_argument("--output", type=str, default=output_dir)
            if config_path is not None:
                parser.add_argument("--config", type=str, default=config_path)
        return parser

    def train(self):
        print(
            f"[{measure_time(self.start_time)}]Loading data on {self.trainer.global_rank}..."
        )
        self.dataModule.setup()
        print(
            f"[{measure_time(self.start_time)}]Data loaded on {self.trainer.global_rank}."
        )

        print(
            f"[{measure_time(self.start_time)}]Initializing model on {self.trainer.global_rank}..."
        )
        model = Transformer(self.config).to(self.device)
        print(
            f"[{measure_time(self.start_time)}]Model initialized on {self.trainer.global_rank}."
        )

        print(
            f"[{measure_time(self.start_time)}]Starting training on {self.trainer.global_rank}..."
        )
        self.trainer.fit(model, self.dataModule)
        print(
            f"[{measure_time(self.start_time)}]Training complete on {self.trainer.global_rank}."
        )

        print(
            f"[{measure_time(self.start_time)}]Starting testing on {self.trainer.global_rank}..."
        )
        self.trainer.test(model, self.dataModule)
        print(
            f"[{measure_time(self.start_time)}]Testing complete on {self.trainer.global_rank}."
        )


