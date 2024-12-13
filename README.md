# How to use

The user must maintain the following steps before running neev for proper functioning.

- Install the module using
```bash
pip install .
```
- All the data must be split into the train, test, val folder of the input directory
- A config of the form must be given in the classes
```json
{
  "input": "./data",
  "output": "./output"
}
```
- In order to run the module use the code
```python
from neev.src import NeevTrainer, DataPreprocess

if __name__ == "__main__":
    preprocess = DataPreprocess("path/to/config.json")
    trainer = NeevTrainer("path/to/config.json")
    preprocess.process()
    trainer.train()
```

## Changing the parameters via config

For full range of tuneable params refer to the config below

```json
{
  "input": "./data",
  "output": "./processed",
  "vocab_size": 32000,
  "num_workers": 4,
  "entries": 8000,

  "batch_size": 4,
  "context_length": 256,
  "embedding_dimension": 128,
  "num_heads": 1,
  "num_layers": 1,
  "dropout": 0.2,

  "learning_rate": 0.0001,
  "weight_decay": 0.01,
  "T_0": 3,
  "T_mult": 2,
  "eta_min": 0.00001,
  "lr_decay": 0.75,

  "max_iterations": 1000,
  "log_interval": 10,
  "eval_interval": 10,
  "precision": 16,

  "deepspeed": {
    "zero_allow_untested_optimizer": true,
    "prescale_gradients": false,
    "zero_optimization": {
      "stage": 2,
      "contiguous_gradients": false,
      "allgather_bucket_size": 5e8,
      "reduce_bucket_size": 5e8,
      "overlap_comm": true
    }
  }
}
```

The config also shows the default values of the params used for training. However not recommended, if the system doesn't allow the use of deepspeed, set the deepspeed param to null.
