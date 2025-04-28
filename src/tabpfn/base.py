"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, overload

import torch

# --- TabPFN imports ---
from tabpfn.constants import (
    AUTOCAST_DTYPE_BYTE_SIZE,
    DEFAULT_DTYPE_BYTE_SIZE,
)
from tabpfn.inference import (
    InferenceEngine,
    InferenceEngineBatchedNoPreprocessing,
    InferenceEngineCacheKV,
    InferenceEngineCachePreprocessing,
    InferenceEngineOnDemand,
)
from tabpfn.model.loading import load_model_criterion_config
from tabpfn.utils import infer_device_and_type, infer_fp16_inference_mode

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    from tabpfn.model.config import InferenceConfig
    from tabpfn.model.transformer import PerFeatureTransformer


class BaseModelSpecs:
    """Base class for model specifications."""

    def __init__(self, model: PerFeatureTransformer, config: InferenceConfig):
        self.model = model
        self.config = config


class ClassifierModelSpecs(BaseModelSpecs):
    """Model specs for classifiers."""

    norm_criterion = None


class RegressorModelSpecs(BaseModelSpecs):
    """Model specs for regressors."""

    def __init__(
        self,
        model: PerFeatureTransformer,
        config: InferenceConfig,
        norm_criterion: FullSupportBarDistribution,
    ):
        super().__init__(model, config)
        self.norm_criterion = norm_criterion


ModelSpecs = Union[RegressorModelSpecs, ClassifierModelSpecs]


@overload
def initialize_tabpfn_model(
    model_path: (str | Path | Literal["auto"] | RegressorModelSpecs),
    which: Literal["regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    static_seed: int,
) -> RegressorModelSpecs: ...


@overload
def initialize_tabpfn_model(
    model_path: (str | Path | Literal["auto"] | ClassifierModelSpecs),
    which: Literal["classifier"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    static_seed: int,
) -> ClassifierModelSpecs: ...


def initialize_tabpfn_model(
    model_path: str
    | Path
    | Literal["auto"]
    | RegressorModelSpecs
    | ClassifierModelSpecs,
    which: Literal["classifier", "regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    static_seed: int,
) -> ModelSpecs:
    """Initializes a TabPFN model based on the provided configuration.

    Args:
        model_path: Path or directive ("auto") to load the pre-trained model from.
        which: Which TabPFN model to load.
        fit_mode: Determines caching behavior.
        static_seed: Random seed for reproducibility logic.

    Returns:
        model: The loaded TabPFN model.
        config: The configuration object associated with the loaded model.
        bar_distribution: The BarDistribution for regression (`None` if classifier).
    """
    # --- 1. Handle Pre-loaded ModelSpecs ---
    if isinstance(model_path, RegressorModelSpecs) and which == "regressor":
        return model_path.model, model_path.config, model_path.norm_criterion
    if isinstance(model_path, ClassifierModelSpecs) and which == "classifier":
        return model_path.model, model_path.config, None
    if model_path is None or isinstance(model_path, (str, Path)):
        # This 'elif' block is intentionally left empty (or could have a 'pass').
        # Its purpose is to identify the valid path types and allow execution
        # to continue to the loading logic below, preventing these types
        # from falling into the final 'else' block.
        pass
    else:
        raise TypeError(
            "Received ModelSpecs via 'model_path', but 'which' parameter is set to '"
            + which
            + "'. Expected 'classifier' or 'regressor'. and model_path"
            + "is of of type"
            + str(type(model_path))
        )

    # (after processing 'auto')
    download = True
    if isinstance(model_path, str) and model_path == "auto":
        model_path = None

    # Load model with potential caching
    if which == "classifier":
        # The classifier's bar distribution is not used;
        # pass check_bar_distribution_criterion=False
        model, _, config_ = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="classifier",
            version="v2",
            download=download,
            model_seed=static_seed,
        )
        bar_distribution = None
    else:
        # The regressor's bar distribution is required
        model, bardist, config_ = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=True,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="regressor",
            version="v2",
            download=download,
            model_seed=static_seed,
        )
        bar_distribution = bardist

    return model, config_, bar_distribution


def determine_precision(
    inference_precision: torch.dtype | Literal["autocast", "auto"],
    device_: torch.device,
) -> tuple[bool, torch.dtype | None, int]:
    """Decide whether to use autocast or a forced precision dtype.

    Args:
        inference_precision:

            - If `"auto"`, decide automatically based on the device.
            - If `"autocast"`, explicitly use PyTorch autocast (mixed precision).
            - If a `torch.dtype`, force that precision.

        device_: The device on which inference is run.

    Returns:
        use_autocast_:
            True if mixed-precision autocast will be used.
        forced_inference_dtype_:
            If not None, the forced precision dtype for the model.
        byte_size:
            The byte size per element for the chosen precision.
    """
    if inference_precision in ["autocast", "auto"]:
        use_autocast_ = infer_fp16_inference_mode(
            device=device_,
            enable=True if (inference_precision == "autocast") else None,
        )
        forced_inference_dtype_ = None
        byte_size = (
            AUTOCAST_DTYPE_BYTE_SIZE if use_autocast_ else DEFAULT_DTYPE_BYTE_SIZE
        )
    elif isinstance(inference_precision, torch.dtype):
        use_autocast_ = False
        forced_inference_dtype_ = inference_precision
        byte_size = inference_precision.itemsize
    else:
        raise ValueError(f"Unknown inference_precision={inference_precision}")

    return use_autocast_, forced_inference_dtype_, byte_size


def create_inference_engine(  # noqa: PLR0913
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: PerFeatureTransformer,
    ensemble_configs: Any,
    cat_ix: list[int],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"],
    device_: torch.device,
    rng: np.random.Generator,
    n_jobs: int,
    byte_size: int,
    forced_inference_dtype_: torch.dtype | None,
    memory_saving_mode: bool | Literal["auto"] | float | int,
    use_autocast_: bool,
    inference_mode: bool = True,
) -> InferenceEngine:
    """Creates the appropriate TabPFN inference engine based on `fit_mode`.

    Each execution mode will perform slightly different operations based on the mode
    specified by the user. In the case where preprocessors will be fit after `prepare`,
    we will use them to further transform the associated borders with each ensemble
    config member.

    Args:
        X_train: Training features
        y_train: Training target
        model: The loaded TabPFN model.
        ensemble_configs: The ensemble configurations to create multiple "prompts".
        cat_ix: Indices of inferred categorical features.
        fit_mode: Determines how we prepare inference (pre-cache or not).
        device_: The device for inference.
        rng: Numpy random generator.
        n_jobs: Number of parallel CPU workers.
        byte_size: Byte size for the chosen inference precision.
        forced_inference_dtype_: If not None, the forced dtype for inference.
        memory_saving_mode: GPU/CPU memory saving settings.
        use_autocast_: Whether we use torch.autocast for inference.
        inference_mode: Whether to use torch.inference_mode (set False if
            backprop is needed)
    """
    engine: (
        InferenceEngineOnDemand
        | InferenceEngineCachePreprocessing
        | InferenceEngineCacheKV
        | InferenceEngineBatchedNoPreprocessing
    )
    if fit_mode == "low_memory":
        engine = InferenceEngineOnDemand.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            ensemble_configs=ensemble_configs,
            rng=rng,
            model=model,
            n_workers=n_jobs,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
        )
    elif fit_mode == "fit_preprocessors":
        engine = InferenceEngineCachePreprocessing.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            ensemble_configs=ensemble_configs,
            n_workers=n_jobs,
            model=model,
            rng=rng,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            inference_mode=inference_mode,
        )
    elif fit_mode == "fit_with_cache":
        engine = InferenceEngineCacheKV.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            model=model,
            ensemble_configs=ensemble_configs,
            n_workers=n_jobs,
            device=device_,
            dtype_byte_size=byte_size,
            rng=rng,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            autocast=use_autocast_,
        )
    elif fit_mode == "batched":
        engine = InferenceEngineBatchedNoPreprocessing.prepare(
            X_trains=X_train,
            y_trains=y_train,
            cat_ix=cat_ix,
            model=model,
            ensemble_configs=ensemble_configs,
            force_inference_dtype=forced_inference_dtype_,
            inference_mode=inference_mode,
            save_peak_mem=memory_saving_mode,
            dtype_byte_size=byte_size,
        )
    else:
        raise ValueError(f"Invalid fit_mode: {fit_mode}")

    return engine


def check_cpu_warning(
    device: str | torch.device, X: np.ndarray | torch.Tensor | pd.DataFrame
) -> None:
    """Check if using CPU with large datasets and warn or error appropriately.

    Args:
        device: The torch device being used
        X: The input data (NumPy array, Pandas DataFrame, or Torch Tensor)
    """
    allow_cpu_override = os.getenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "0") == "1"
    device_mapped = infer_device_and_type(device)

    # Determine number of samples
    try:
        num_samples = X.shape[0]
    except AttributeError:
        return

    if torch.device(device_mapped).type == "cpu":
        if num_samples > 1000:
            if not allow_cpu_override:
                raise RuntimeError(
                    "Running on CPU with more than 1000 samples is not allowed "
                    "by default due to slow performance.\n"
                    "To override this behavior, set the environment variable "
                    "TABPFN_ALLOW_CPU_LARGE_DATASET=1.\n"
                    "Alternatively, consider using a GPU or the tabpfn-client API: "
                    "https://github.com/PriorLabs/tabpfn-client"
                )
        elif num_samples > 200:
            warnings.warn(
                "Running on CPU with more than 200 samples may be slow.\n"
                "Consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client",
                stacklevel=2,
            )
