"""
CatalogEditor: Core Inpainting and CLIPSeg Auto-masking Engine.

Handles text-driven open-vocabulary segmentation with CIDAS/clipseg-rd64-refined
and specification inpainting with stable-diffusion-v1-5/stable-diffusion-inpainting.
"""

from typing import Dict, Optional
import numpy as np
from PIL import Image, ImageFilter

# Safe top-level imports: avoid importing heavy frameworks at import time
# to ensure fast and error-free CPU importing on development environments.
try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import StableDiffusionInpaintPipeline
except ImportError:
    StableDiffusionInpaintPipeline = None

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
except ImportError:
    CLIPSegProcessor = None
    CLIPSegForImageSegmentation = None


def paste_back(
    original: Image.Image,
    inpainted: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Composites the inpainted image onto the original image using mask as alpha mask.
    Pixels where mask == 0 remain strictly byte-identical to original.

    Args:
        original: Original unedited PIL Image.
        inpainted: Inpainted PIL Image.
        mask: Mask PIL Image (mode 'L'). Non-zero indicates edited region.

    Returns:
        PIL Image with pristine unmasked original pixels and inpainted edit region.
    """
    orig_rgb = original.convert("RGB")
    inpaint_rgb = inpainted.convert("RGB")
    mask_l = mask.convert("L")

    if inpaint_rgb.size != orig_rgb.size:
        inpaint_rgb = inpaint_rgb.resize(orig_rgb.size, Image.Resampling.LANCZOS)
    if mask_l.size != orig_rgb.size:
        mask_l = mask_l.resize(orig_rgb.size, Image.Resampling.BILINEAR)

    final_image = orig_rgb.copy()
    final_image.paste(inpaint_rgb, (0, 0), mask_l)
    return final_image


class CatalogEditor:
    """
    Catalog image editor using zero-shot segmentation (CLIPSeg) and
    Stable Diffusion Inpainting (v1.5) for furniture spec updates.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the editor with a target device. Models are lazily loaded
        only upon first execution to allow instant, error-free module imports.
        """
        # Automatically fall back to CPU if CUDA is requested but unavailable
        if device == "cuda" and (torch is None or not torch.cuda.is_available()):
            self.device = "cpu"
        else:
            self.device = device

        self._clipseg_processor = None
        self._clipseg_model = None
        self._inpaint_pipe = None

    def _load_clipseg(self):
        """Lazily loads the CLIPSeg segmentation processor and model."""
        if self._clipseg_model is not None and self._clipseg_processor is not None:
            return

        global torch, CLIPSegProcessor, CLIPSegForImageSegmentation
        if torch is None:
            import torch
        if CLIPSegProcessor is None or CLIPSegForImageSegmentation is None:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        model_id = "CIDAS/clipseg-rd64-refined"
        self._clipseg_processor = CLIPSegProcessor.from_pretrained(model_id)
        self._clipseg_model = CLIPSegForImageSegmentation.from_pretrained(model_id)
        self._clipseg_model.to(self.device)
        self._clipseg_model.eval()

    def _load_inpaint(self):
        """Lazily loads the Stable Diffusion Inpainting pipeline."""
        if self._inpaint_pipe is not None:
            return

        global torch, StableDiffusionInpaintPipeline
        if torch is None:
            import torch
        if StableDiffusionInpaintPipeline is None:
            from diffusers import StableDiffusionInpaintPipeline

        model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        self._inpaint_pipe.to(self.device)
        self._inpaint_pipe.enable_attention_slicing()

    def generate_mask(
        self,
        image: Image.Image,
        part_text: str,
        threshold: float = 0.4,
        dilate_px: int = 12,
    ) -> Image.Image:
        """
        Generates a binary mask identifying the targeted furniture component.

        Args:
            image: Input PIL Image.
            part_text: Text description of the component (e.g. 'chair legs', 'cushion').
            threshold: Sigmoid probability cutoff for mask binarization.
            dilate_px: Pixel radius for morphological dilation.

        Returns:
            PIL Image in mode 'L' (grayscale), where white (255) indicates the edit region.
        """
        self._load_clipseg()

        orig_w, orig_h = image.size
        img_rgb = image.convert("RGB")

        inputs = self._clipseg_processor(
            text=[part_text],
            images=[img_rgb],
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._clipseg_model(**inputs)
            preds = outputs.logits  # Shape: [1, H, W]

        # Apply sigmoid to convert logits to probabilities
        preds_2d = preds[0] if preds.ndim == 3 else preds.squeeze()
        probs = torch.sigmoid(preds_2d).cpu().float().numpy()

        # Binary thresholding: white (255) = edit region, black (0) = keep
        binary_mask = (probs > threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary_mask, mode="L")

        # Resize to original image dimensions
        mask_img = mask_img.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

        # Morphological dilation using MaxFilter
        if dilate_px > 0:
            filter_size = max(3, dilate_px * 2 + 1)
            mask_img = mask_img.filter(ImageFilter.MaxFilter(filter_size))

        # Light Gaussian blur to feather mask edges for smooth blending
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))

        return mask_img.convert("L")

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 30,
        guidance: float = 7.5,
        strength: float = 1.0,
        seed: int = 42,
    ) -> Image.Image:
        """
        Applies inpainting to the masked region while ensuring all pixels outside
        the mask remain strictly identical to the original image.

        Args:
            image: Original PIL Image.
            mask: Binary or feathered PIL Image in mode 'L' (white = edit area).
            prompt: Text prompt describing the target specification.
            negative_prompt: Negative guidance prompt.
            steps: Number of diffusion inference steps.
            guidance: Classifier-free guidance scale.
            strength: Inpainting conditioning strength.
            seed: Random generator seed for reproducibility.

        Returns:
            Inpainted PIL Image with original unmasked pixels preserved bit-for-bit.
        """
        self._load_inpaint()

        orig_w, orig_h = image.size
        img_rgb = image.convert("RGB")
        mask_l = mask.convert("L")

        # Pad to square with white background to fit 512x512 without aspect distortion
        max_dim = max(orig_w, orig_h)
        pad_x = (max_dim - orig_w) // 2
        pad_y = (max_dim - orig_h) // 2

        square_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        square_img.paste(img_rgb, (pad_x, pad_y))

        # For mask, pad with black (0 = unedited/keep)
        square_mask = Image.new("L", (max_dim, max_dim), 0)
        square_mask.paste(mask_l, (pad_x, pad_y))

        # Resize to 512x512 for Stable Diffusion
        inpaint_input_img = square_img.resize((512, 512), Image.Resampling.LANCZOS)
        inpaint_input_mask = square_mask.resize((512, 512), Image.Resampling.BILINEAR)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Execute diffusion inpainting
        inpaint_output = self._inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=inpaint_input_img,
            mask_image=inpaint_input_mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            generator=generator,
        ).images[0]

        # Resize back to square dimension
        result_square = inpaint_output.resize((max_dim, max_dim), Image.Resampling.LANCZOS)

        # Crop back to original dimensions
        result_cropped = result_square.crop((pad_x, pad_y, pad_x + orig_w, pad_y + orig_h))

        # Paste result ONLY inside the mask region onto the pristine original image
        final_image = paste_back(img_rgb, result_cropped, mask_l)

        # Clear intermediate diffusion tensors to keep VRAM footprint minimal
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        return final_image

    def edit(
        self,
        image: Image.Image,
        part_text: str,
        new_spec_prompt: str,
        negative_prompt: str = "",
        threshold: float = 0.4,
        dilate_px: int = 12,
        steps: int = 30,
        guidance: float = 7.5,
        strength: float = 1.0,
        seed: int = 42,
        mask: Optional[Image.Image] = None,
        **kwargs,
    ) -> Dict[str, Image.Image]:
        """
        Executes end-to-end auto-masking and inpainting for a single image.

        Args:
            image: Input product catalog image.
            part_text: Description of part to mask (e.g. 'chair legs').
            new_spec_prompt: Prompt for new material/specification.
            negative_prompt: Unwanted attributes prompt.
            threshold: Mask probability threshold.
            dilate_px: Mask dilation radius.
            steps: Inpainting diffusion steps.
            guidance: Guidance scale.
            strength: Inpainting strength.
            seed: Random seed.
            mask: Optional pre-computed or manual mask override.

        Returns:
            Dict containing 'mask' and 'result' PIL Images.
        """
        if mask is None:
            mask = self.generate_mask(
                image=image,
                part_text=part_text,
                threshold=threshold,
                dilate_px=dilate_px,
            )

        result = self.inpaint(
            image=image,
            mask=mask,
            prompt=new_spec_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            strength=strength,
            seed=seed,
        )

        return {"mask": mask, "result": result}

