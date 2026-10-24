"""Edit Python settings here; each condition runs baseline and PF-HMM at both seeds."""

from pfhmm_vision.config import configure


SEEDS = (0, 2)
DEVICE = "cuda"
RUNS_DIR = "runs/cifar100_study"

BASE_CONFIG = configure(
    data={"root": "data", "workers": 4},
    model={"architecture": "resnet18", "adapter_hidden": 128},
    pretrain={"epochs": 200, "batch_size": 128, "learning_rate": 0.1, "warmup_epochs": 5},
    calibration={"steps": 150, "learning_rate": 0.03},
    bandit={"learning_rate": 1e-4, "batch_size": 64, "kl_coefficient": 0.1,
            "entropy_coefficient": 0.001, "checkpoint_every": 25, "log_every": 10,
            "phases": [
                {"name": "stable_reliable", "updates": 125, "volatility": 0.002, "feedback_flip": 0.05},
                {"name": "stable_noisy", "updates": 125, "volatility": 0.002, "feedback_flip": 0.30},
                {"name": "volatile_reliable", "updates": 125, "volatility": 0.02, "feedback_flip": 0.05},
                {"name": "volatile_noisy", "updates": 125, "volatility": 0.02, "feedback_flip": 0.30},
            ]},
    filter={"particles": 512},
    evaluation={"batch_size": 256},
)

CONDITIONS = {"default": {}}
