# PF-HMM vision post-training pilot

This project implements an experimental, action-conditioned extension of a human
uncertainty model. It is not the authors' original task implementation, a certified
novel algorithm, or an established accuracy improvement.

The motivating paper is Fang and Piray (2026), *Inferring the causes of noise from
binary outcomes: A normative theory of learning under uncertainty*,
[Psychological Review](https://doi.org/10.1037/rev0000638). Its abstract describes
inference that can "simultaneously infer volatility and stochasticity".
The original task uses a binary hidden state and binary outcomes. Here, the hidden
state indexes two visual corruption contexts; a separate calibration model supplies
input- and action-conditioned correctness likelihoods. This is our extension.

## Run

Use a Linux NVIDIA server for CUDA experiments. From the extracted project folder:

```text
python3 launch.py --device cuda --seeds 0 2
```

This creates `.conda` with Python 3.11, installs PyTorch 2.8.0 / torchvision 0.23.0
from the official CUDA 12.8 wheel index, verifies real convolution gradients,
downloads CIFAR-100, and runs baseline and PF-HMM at seeds 0 and 2 (four arms).
It trains an independent source classifier per seed and shares it between the two
methods within that seed.
An installed compatible NVIDIA driver is required; a separate CUDA toolkit is not
required by the wheel recipe. The project uses no venv or shell scripts.

The separate setup and experiment entry points are:

```text
python3 bootstrap.py --device cuda
./.conda/bin/python -s experiment.py --device cuda --seeds 0 2
```

The installer needs an existing Conda executable. If it is not on PATH, pass
`--conda /absolute/path/to/conda`. It prints installation stages and elapsed time
every 30 seconds while Conda or pip is running. Download/solver ETA is explicitly
unknown; pip and Conda retain their own live download output.

## Python experiment settings

`configs/experiment.py` contains the seeds, device, output directory, common
hyperparameters, and a dictionary of conditions. Each condition automatically runs
both methods for both configured seeds. No training implementation edits are needed.
The default is one condition with seeds `(0, 2)`.

For a Python hyperparameter grid, change only the settings in that file:

```python
from itertools import product
from pfhmm_vision.config import configure

SEEDS = (0, 2)
DEVICE = "cuda"
RUNS_DIR = "runs/lr_kl_grid"
BASE_CONFIG = configure(
    pretrain={"epochs": 200, "batch_size": 128, "learning_rate": 0.1},
    bandit={"batch_size": 64, "entropy_coefficient": 0.001},
    filter={"particles": 512},
)
CONDITIONS = {
    f"lr_{lr:g}_kl_{kl:g}": {
        "bandit": {"learning_rate": lr, "kl_coefficient": kl}
    }
    for lr, kl in product((1e-4, 3e-4), (0.05, 0.1))
}
```

This example requests 4 conditions x 2 seeds x 2 methods = 16 adaptation runs.
Unknown fields and invalid budgets are rejected before training. Both methods use
the same resolved configuration within a condition. Matching source pretraining is
reused across compatible conditions separately for each seed. Different source
recipes, splits, or architectures require separate source training.

Inspect the plan without installing dependencies, downloading data, or training:

```text
python3 experiment.py --dry-run
```

You can also call the runner directly from Python:

```python
from pfhmm_vision.config import configure
from pfhmm_vision.study import run_study

config = configure(bandit={"learning_rate": 1e-4}, filter={"particles": 512})
conditions = {"default": {}}
run_study(config, conditions, seeds=(0, 2), root="runs/my_study", device="cuda")
```

CLI/settings paths resolve against the project directory, including invocation from
its parent. Direct Python API paths resolve against the working directory.
Re-running resumes committed work. Different resolved configurations or seeds receive
separate case directories, so changing a condition does not overwrite an earlier
result. Adding or reordering conditions does not invalidate unchanged cases. Python
settings are recorded through their full resolved configurations and are excluded
from the training-source hash. Changing training source still requires a new
`--runs-dir`. Partial policy checkpoints require the same numerical runtime.

## Progress and timing

Source pretraining, feature extraction, temperature fitting, each method, and final
evaluation print completed/total work, percentage, elapsed time, and an ETA estimated
from measured throughput. Source training also reports epoch and batch; adaptation
reports updates and consumed feedback. Long stages print at least once per 15 seconds
when a batch/update finishes, and adaptation additionally follows `bandit.log_every`.
The first ETA is `pending` until work has been measured. Resumed source/adapter stages
restore their committed elapsed time. Seed/condition counters also show progress and
invocation elapsed time; their ETAs use completed seeds/conditions and are coarse.

Representative format (illustrative numbers, not a GPU benchmark):

```text
pretrain batches 549/54900 (1.0%) elapsed=45s ETA=1h 14m 15s epoch=2/200 batch=264/274
pfhmm updates 100/500 (20.0%) elapsed=12s ETA=48s feedback=6400 observed_reward=0.5000 kl=0.00100
```

No fixed total-hours estimate is assumed before measuring the actual GPU. A seed's
`complete.json` stores invocation duration and source checkpoint provenance; each
method's `training_complete.json` stores cumulative training duration and peak CUDA
allocation. Suite and study summaries store invocation duration, and `progress.json`
stores the latest completed seed/condition counter.

## Default experiment

- CIFAR-100 training images: 35,000 source-pretraining, 5,000 calibration,
  7,500 adaptation, and 2,500 validation images, stratified and disjoint.
- One CIFAR-stem ResNet-18 per seed is trained for 200 epochs. The final source epoch
  is used. A completed matching `shared/pretrain.pt` can be reused using
  `--base-checkpoint`; arbitrary third-party checkpoints are intentionally not accepted
  because their source split and label access cannot be verified by this loader.
  The Python study runner automatically supplies compatible completed checkpoints
  across conditions and persists the established source before cache/training work.
- Source normalization uses source-pretraining pixels only. Calibration fits one
  temperature per configured visual context. Both arms have the same calibration
  information and action-independent control variate.
- Frozen backbone, identical zero-initialized 128-unit residual adapters.
  Frozen features use one deterministic corruption view per image and context.
  They are cached to reduce GPU compute and stored without adaptation labels.
- `baseline`: on-policy REINFORCE with observed bits, categorical KL to the
  source policy, and entropy regularization. This is a controlled baseline, not a
  claim to implement DRC or the strongest noisy-reward RL algorithm.
- `pfhmm`: the same optimizer and policy objective, replacing the observed bit by
  posterior clean correctness from a joint particle filter. The posterior reward
  estimator is model-dependent; it is not guaranteed unbiased under misspecification.
- Four phases independently vary context-switch probability and feedback flips.
  The learner does not receive phase boundaries, true context, or true noise rates.
  Each arm consumes exactly 500 updates x 64 observations = 32,000 feedback bits.
  The experiment repeats at seeds 0 and 2. Within each seed both methods use identical initial adapter parameters, image indices,
  context transitions, action-sampling uniforms, and feedback-flip uniforms.
  Actions and rewards can differ after policies diverge.
- Ground truth for mechanism diagnostics is released only after an arm completes.
  Validation is logged without selecting a checkpoint. Official test examples are
  evaluated only after all requested arms finish; they never enter calibration or
  post-training. Results use the fixed final update.
- Final tests: clean, Gaussian noise 0.15, Gaussian noise 0.25, and contrast 0.3.
  These are this project's deterministic corruptions, not CIFAR-100-C results.

The 24GB target is supported by the small frozen-backbone adapter design. Actual
CUDA memory and throughput must be measured on the user's GPU. Source pretraining
is the substantial initial cost; it is performed once per seed for a compatible
source recipe and shared between methods and compatible conditions.

## Inspect results

`runs/cifar100_study/summary.json` links the condition summaries. Each condition has
its own directory under `runs/cifar100_study/<condition>/<configuration-id>/`, with
`summary.csv` and `summary.json` reporting across-seed means, sample standard
deviations, and paired PF-HMM-minus-baseline gains. Historical study plans and
summaries are retained under `studies/<study-id>/`.

Inside a condition directory, `seed0/results.csv` and `seed0/results.json` (and
the corresponding `seed2/` files) contain final accuracy, NLL,
ECE, Brier score, and PF-HMM-minus-baseline accuracy in percentage points.
`test_predictions.npz` retains logits and labels for independent recomputation.
Each arm stores `checkpoint.pt`, `history.json`, `training_complete.json`, and
`feedback_audit.npz`. `shared/likelihood_diagnostics.json` reports the separation
between the two context-conditioned likelihoods. Similar likelihoods can make the
causes unidentifiable. A gain is not assumed; failures and degradation are retained.

Two seeds remain a descriptive pilot. Compare reward
inference errors and context/noise diagnostics alongside classification results.

## Verify locally

Run these using the Conda interpreter:

```text
python -m unittest discover -s tests -v
python smoke.py --device cpu
```

The smoke test exercises actual ResNet source training, caches, calibration,
both RL arms, checkpoint reuse, and final-test output on synthetic images. It
provides execution evidence only. No real CIFAR-100 efficacy or target-GPU result
is implied by a synthetic smoke pass.

The default experiment settings are Python in `configs/experiment.py`; a strict
JSON base remains available in `configs/cifar100.json` using `--config`. The original
single-seed command is `python run_experiment.py --device cuda --seed 0`.
Data, backbone, adapter,
feedback environment, reward processors, particle inference, training objective,
checkpointing, calibration, and evaluation are separate modules. Changing the data
provider or backbone requires implementing that interface rather than renaming a
configuration to an unsupported architecture.
