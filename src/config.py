"""
CatalogUpdate AI - Global Configuration & Default Tuning Parameters.

Centralizes all default hyperparameters for segmentation, inpainting,
aspect ratio padding, and batch execution.
"""

from typing import Tuple

# Inpainting & Diffusion Hyperparameters
DEFAULT_STEPS: int = 30
DEFAULT_GUIDANCE: float = 7.5
DEFAULT_STRENGTH: float = 1.0
DEFAULT_SEED: int = 42
INPAINT_SIZE: Tuple[int, int] = (512, 512)
PADDING_COLOR: Tuple[int, int, int] = (255, 255, 255)

# Segmentation & Auto-Masking Hyperparameters
DEFAULT_MASK_THRESHOLD: float = 0.4
DEFAULT_RETRY_MASK_THRESHOLD: float = 0.2  # Sensitive fallback threshold for thin/angled components
DEFAULT_MASK_DILATION_PX: int = 12
DEFAULT_MASK_BLUR_RADIUS: int = 2
MIN_MASK_COVERAGE_PCT: float = 0.5  # Percentage of image area below which mask is considered absent

# ControlNet Structure Preservation Hyperparameters
DEFAULT_USE_CONTROLNET: bool = False
DEFAULT_CONTROLNET_SCALE: float = 0.8
DEFAULT_CANNY_LOW: int = 100
DEFAULT_CANNY_HIGH: int = 200

# Model Identifiers
INPAINT_MODEL_ID: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
CLIPSEG_MODEL_ID: str = "CIDAS/clipseg-rd64-refined"
CONTROLNET_MODEL_ID: str = "lllyasviel/control_v11p_sd15_canny"
