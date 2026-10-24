"""Strict configuration schemas shared by data, inference, and training components."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class View:
    kind: str = "clean"
    severity: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    name: str = "cifar100"
    root: str = "data"
    pretrain_per_class: int = 350
    calibration_per_class: int = 50
    adaptation_per_class: int = 75
    validation_per_class: int = 25
    split_seed: int = 20261024
    workers: int = 4
    download: bool = True
    contexts: tuple = (View(), View("gaussian_noise", 0.15))
    test_views: tuple = (
        View(), View("gaussian_noise", 0.15),
        View("gaussian_noise", 0.25), View("contrast", 0.3),
    )


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "resnet18"
    small_stem: bool = True
    adapter_hidden: int = 128


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    gradient_clip: float = 10.0


@dataclass(frozen=True)
class CalibrationConfig:
    steps: int = 150
    learning_rate: float = 0.03
    minimum_temperature: float = 0.25
    maximum_temperature: float = 10.0


@dataclass(frozen=True)
class Phase:
    name: str = "stable_reliable"
    updates: int = 125
    volatility: float = 0.002
    feedback_flip: float = 0.05


@dataclass(frozen=True)
class BanditConfig:
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    kl_coefficient: float = 0.1
    entropy_coefficient: float = 0.001
    gradient_clip: float = 1.0
    checkpoint_every: int = 25
    log_every: int = 10
    phases: tuple = (
        Phase(), Phase("stable_noisy", 125, 0.002, 0.30),
        Phase("volatile_reliable", 125, 0.02, 0.05),
        Phase("volatile_noisy", 125, 0.02, 0.30),
    )

    @property
    def updates(self):
        """Return the total update count without maintaining a second budget field."""
        return sum(phase.updates for phase in self.phases)


@dataclass(frozen=True)
class FilterConfig:
    particles: int = 512
    volatility_initial: float = 0.01
    noise_initial: float = 0.1
    volatility_diffusion: float = 0.001
    noise_diffusion: float = 0.001
    resample_fraction: float = 0.5
    probability_floor: float = 1e-6


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 256
    ece_bins: int = 15


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self):
        """Reject invalid budgets and unsupported generative assumptions before execution."""
        _validate_numbers(self.as_dict())
        integral_values = [self.data.pretrain_per_class, self.data.calibration_per_class,
                           self.data.adaptation_per_class, self.data.validation_per_class,
                           self.data.split_seed, self.data.workers, self.model.adapter_hidden,
                           self.pretrain.epochs, self.pretrain.batch_size, self.pretrain.warmup_epochs,
                           self.calibration.steps, self.bandit.batch_size,
                           self.bandit.checkpoint_every, self.bandit.log_every, self.filter.particles,
                           self.evaluation.batch_size, self.evaluation.ece_bins]
        integral_values.extend(p.updates for p in self.bandit.phases)
        if any(type(value) is not int for value in integral_values):
            raise ValueError("Counts, seeds, and intervals must be integers")
        if self.data.split_seed < 0:
            raise ValueError("The split seed cannot be negative")
        if self.data.name != "cifar100":
            raise ValueError("Only the CIFAR-100 data provider is currently registered")
        sizes = [self.data.pretrain_per_class, self.data.calibration_per_class,
                 self.data.adaptation_per_class, self.data.validation_per_class]
        if any(not isinstance(n, int) or n <= 0 for n in sizes) or sum(sizes) > 500:
            raise ValueError("Positive split counts must fit the 500 training examples per class")
        if len(self.data.contexts) != 2:
            raise ValueError("The binary-state PF-HMM extension requires exactly two contexts")
        if len({digest(asdict(view)) for view in self.data.contexts}) != 2:
            raise ValueError("The two latent contexts must be distinct")
        if len({digest(asdict(view)) for view in self.data.test_views}) != len(self.data.test_views):
            raise ValueError("Duplicate test views would overwrite prediction conditions")
        for view in (*self.data.contexts, *self.data.test_views):
            if view.kind not in ("clean", "gaussian_noise", "contrast"):
                raise ValueError(f"Unknown view kind: {view.kind}")
            if view.severity < 0 or (view.kind == "contrast" and view.severity > 1):
                raise ValueError("Invalid corruption severity")
            if view.kind == "clean" and view.severity != 0:
                raise ValueError("A clean view must have severity zero")
        if not self.data.test_views or self.data.workers < 0:
            raise ValueError("Evaluation views are required and workers must be nonnegative")
        if self.model.architecture not in ("resnet18", "resnet34"):
            raise ValueError("Unknown backbone architecture")
        positives = [self.model.adapter_hidden, self.pretrain.epochs, self.pretrain.batch_size,
                     self.pretrain.learning_rate, self.calibration.steps,
                     self.pretrain.gradient_clip,
                     self.calibration.learning_rate, self.bandit.batch_size,
                     self.bandit.learning_rate, self.bandit.gradient_clip,
                     self.bandit.checkpoint_every, self.bandit.log_every,
                     self.evaluation.batch_size, self.evaluation.ece_bins]
        if any(value <= 0 for value in positives):
            raise ValueError("Sizes, learning rates, and intervals must be positive")
        if not 0 <= self.pretrain.warmup_epochs < self.pretrain.epochs:
            raise ValueError("Warmup must be shorter than pretraining")
        if any(value < 0 for value in (self.bandit.kl_coefficient,
                                     self.bandit.entropy_coefficient,
                                     self.bandit.weight_decay, self.pretrain.weight_decay)):
            raise ValueError("Penalty coefficients cannot be negative")
        if not 0 <= self.pretrain.momentum < 1:
            raise ValueError("SGD momentum must be in [0, 1)")
        if not self.bandit.phases or any(p.updates <= 0 for p in self.bandit.phases):
            raise ValueError("Every phase requires a positive update count")
        if any(not 0 <= p.volatility < 0.5 or not 0 <= p.feedback_flip < 0.5
               for p in self.bandit.phases):
            raise ValueError("Volatility and flip rates must be in [0, .5)")
        if self.filter.particles < 2 or not 0 < self.filter.resample_fraction <= 1:
            raise ValueError("Invalid particle count or resampling threshold")
        if not 0 < self.filter.probability_floor < 0.01:
            raise ValueError("Invalid numerical probability floor")
        if any(not 0 < value < 0.5 for value in
               (self.filter.volatility_initial, self.filter.noise_initial)):
            raise ValueError("Initial filter probabilities must be in (0, .5)")
        if min(self.filter.volatility_diffusion, self.filter.noise_diffusion) < 0:
            raise ValueError("Particle diffusion cannot be negative")
        if not 0 < self.calibration.minimum_temperature < self.calibration.maximum_temperature:
            raise ValueError("Invalid calibration temperature bounds")
        return self

    def as_dict(self):
        """Export the complete resolved configuration for provenance."""
        return asdict(self)

    def fingerprint(self):
        """Identify all experimental parameters using canonical JSON."""
        return digest(self.as_dict())


def digest(value):
    """Hash a JSON-serializable object without depending on dictionary insertion order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_numbers(value):
    """Reject JSON NaN and infinity before they can bypass probability-bound comparisons."""
    if isinstance(value, dict):
        for child in value.values():
            _validate_numbers(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _validate_numbers(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Configuration numbers must be finite")


def _merge(default, values):
    """Build nested immutable schemas while rejecting misspelled configuration fields."""
    if not isinstance(values, dict):
        raise ValueError("Configuration sections must be JSON objects")
    allowed = {f.name for f in fields(default)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown fields for {type(default).__name__}: {sorted(unknown)}")
    resolved = {}
    for key, value in values.items():
        old = getattr(default, key)
        if is_dataclass(old):
            resolved[key] = _merge(old, value)
        elif key in ("contexts", "test_views"):
            resolved[key] = tuple(_merge(View(), item) for item in value)
        elif key == "phases":
            resolved[key] = tuple(_merge(Phase(), item) for item in value)
        else:
            resolved[key] = value
    return type(default)(**{**as_field_values(default), **resolved})


def as_field_values(value):
    """Keep nested dataclass objects intact when reconstructing one section."""
    return {f.name: getattr(value, f.name) for f in fields(value)}


def load_config(path=None):
    """Load a partial JSON configuration and expand all documented defaults."""
    values = {} if path is None else json.loads(Path(path).read_text())
    return configure(**values)


def configure(base=None, **overrides):
    """Apply strict nested Python dictionaries to an immutable base configuration."""
    if base is not None and not isinstance(base, Config):
        raise TypeError("base must be a Config instance")
    return _merge(Config() if base is None else base, overrides).validate()
