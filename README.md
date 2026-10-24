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
python3 launch.py --device cuda --seed 0
```

This creates `.conda` with Python 3.11, installs PyTorch 2.8.0 / torchvision 0.23.0
from the official CUDA 12.8 wheel index, verifies real convolution gradients,
downloads CIFAR-100, trains one shared source classifier, and runs both arms once.
An installed compatible NVIDIA driver is required; a separate CUDA toolkit is not
required by the wheel recipe. The project uses no venv or shell scripts.

The separate setup and experiment entry points are:

```text
python3 bootstrap.py --device cuda
conda run --no-capture-output --prefix ./.conda python run_experiment.py --device cuda --seed 0
```

All launcher paths resolve against the project directory, including when launched
from its parent. Re-running the same command resumes committed work. A run directory
rejects different source, configuration, method selection, or seed. Choose another
`--runs-dir` after changing a configuration. Partial policy checkpoints require the
same numerical runtime to continue exactly.

## Default experiment

- CIFAR-100 training images: 35,000 source-pretraining, 5,000 calibration,
  7,500 adaptation, and 2,500 validation images, stratified and disjoint.
- One shared CIFAR-stem ResNet-18 is trained for 200 epochs. The final source epoch
  is used. A completed matching `shared/pretrain.pt` can be reused using
  `--base-checkpoint`; arbitrary third-party checkpoints are intentionally not accepted
  because their source split and label access cannot be verified by this loader.
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
  Both use seed 0 by default, identical initial adapter parameters, image indices,
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
is the substantial initial cost; it is performed once, not once per arm.

## Inspect results

`runs/cifar100_seed0/results.csv` and `results.json` contain final accuracy, NLL,
ECE, Brier score, and PF-HMM-minus-baseline accuracy in percentage points.
`test_predictions.npz` retains logits and labels for independent recomputation.
Each arm stores `checkpoint.pt`, `history.json`, `training_complete.json`, and
`feedback_audit.npz`. `shared/likelihood_diagnostics.json` reports the separation
between the two context-conditioned likelihoods. Similar likelihoods can make the
causes unidentifiable. A gain is not assumed; failures and degradation are retained.

One seed is a pilot, not evidence of across-seed superiority. Compare reward
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

Configuration is strict JSON in `configs/cifar100.json`. Data, backbone, adapter,
feedback environment, reward processors, particle inference, training objective,
checkpointing, calibration, and evaluation are separate modules. Changing the data
provider or backbone requires implementing that interface rather than renaming a
configuration to an unsupported architecture.
