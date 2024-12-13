import torch
from .data import DataModule
from .model import Transformer
from .utils import measure_time, ConfigReader

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


class NeevTrainer:
    def __init__(self, config_path):
        torch.set_float32_matmul_precision("high")
        self.config = ConfigReader(config_path).read()
        self.start_time = measure_time()
        self.lr_monitor = LearningRateMonitor(logging_interval="step")
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

        self.dataModule = DataModule(self.config)
        if self.config.deepspeed is not None:
            strategy = DeepSpeedStrategy(config=self.config.deepspeed)
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
            devices=self.config.devices,
            max_steps=self.config.max_iterations,
            precision=self.config.precision,
            log_every_n_steps=self.config.log_interval,
            val_check_interval=self.config.eval_interval,
            strategy=strategy,
            logger=self.logger,
            profiler=self.profiler,
            callbacks=[self.lr_monitor, self.checkpoint],
            gradient_clip_val=self.config.gradient_clip_val,
        )

        self.model = Transformer(self.config).to(self.device)

    def train(self):
        print(
            f"[{measure_time(self.start_time)}]Loading data on {self.trainer.global_rank}..."
        )
        self.dataModule.setup()
        print(
            f"[{measure_time(self.start_time)}]Data loaded on {self.trainer.global_rank}."
        )

        print(
            f"[{measure_time(self.start_time)}]Starting training on {self.trainer.global_rank}..."
        )
        self.trainer.fit(self.model, self.dataModule)
        print(
            f"[{measure_time(self.start_time)}]Training complete on {self.trainer.global_rank}."
        )

        print(
            f"[{measure_time(self.start_time)}]Starting testing on {self.trainer.global_rank}..."
        )
        self.trainer.test(self.model, self.dataModule)
        print(
            f"[{measure_time(self.start_time)}]Testing complete on {self.trainer.global_rank}."
        )
