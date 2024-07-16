from .data import DataModule
from .model import Transformer
from .utils.misc import measure_time
from .data.tokenizer import Tokenizer
from .data.tensorizer import Tensorizer
from .utils.args_parser import training_parser

import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class NeevTrainer:
    def __init__(self, input_dir=None, output_dir=None, config_path=None, parser=None):

        parser = self.check_args(input_dir, output_dir, config_path, parser)
        torch.set_float32_matmul_precision("high")
        self.lr_monitor = LearningRateMonitor(logging_interval="step")
        config = training_parser(parser)
        self.config = config
        self.tokenizer = Tokenizer(
            config.input,
            config.output,
            config.vocab_size,
            config.spm_convergence,
            config.tokenizer,
        )
        self.tensorizer = Tensorizer(
            config.vocab_size, config.context_length, config.output
        )
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
        if config.deepspeed is not None:
            strategy = DeepSpeedStrategy(config=config.deepspeed)
        else:
            strategy = "ddp"
        self.dataModule = DataModule(config, self.tokenizer, self.tensorizer)
        self.checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"logs/checkpoints/",
            filename="best-checkpoint",
            save_top_k=1,
            mode="min",
        )

        self.trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=config.max_epochs,
            min_epochs=config.min_epochs,
            precision=config.precision,
            log_every_n_steps=config.log_interval,
            strategy=strategy,
            logger=self.logger,
            profiler=self.profiler,
            callbacks=[self.lr_monitor, self.checkpoint],
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


if __name__ == "__main__":
    parser = ArgumentParser()
    NeevTrainer(parser=parser).train()
