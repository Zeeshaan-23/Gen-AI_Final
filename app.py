"""
CatalogUpdate AI - Gradio Web Interface.

Features:
- Tab 1: Batch Catalog Update (multi-image upload, structured spec fields, preset dropdowns, manifest & zip downloads).
- Tab 2: Single Image with interactive manual/auto masking (gr.ImageEditor + CLIPSeg).
"""

import os
import sys
import tempfile
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import gradio as gr

from src.config import (
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_RETRY_MASK_THRESHOLD,
    DEFAULT_MASK_DILATION_PX,
    DEFAULT_SEED,
    DEFAULT_USE_CONTROLNET,
    DEFAULT_CONTROLNET_SCALE,
)
from src.prompts import FURNITURE_PRESETS, build_prompt, DEFAULT_NEGATIVE_PROMPT
from src.engine import CatalogEditor
from src.batch import run_batch, zip_outputs, MockCatalogEditor

# Determine device and load CatalogEditor once at application startup
# Model weights remain lazily deferred until inference is executed.
is_mock_env = (
    "--mock" in sys.argv
    or os.environ.get("USE_MOCK_ENGINE", "0").strip().lower() in ("1", "true", "yes")
)
if is_mock_env:
    editor = MockCatalogEditor(device="cpu")
else:
    editor = CatalogEditor(device="cuda")


def is_colab() -> bool:
    """Detects if running inside Google Colab."""
    if "google.colab" in sys.modules:
        return True
    try:
        import importlib
        importlib.import_module("google.colab")
        return True
    except ImportError:
        return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


def extract_mask_from_editor(editor_data):
    """
    Extracts background image and painted mask from gr.ImageEditor dict.
    Returns: (PIL.Image or None, PIL.Image or None).
    """
    if editor_data is None:
        return None, None

    if isinstance(editor_data, dict):
        bg = editor_data.get("background")
        layers = editor_data.get("layers", [])
        if bg is None:
            return None, None

        bg_img = bg.convert("RGB")
        if layers and len(layers) > 0:
            combined = layers[0].convert("RGBA")
            for layer in layers[1:]:
                combined = Image.alpha_composite(combined, layer.convert("RGBA"))

            alpha_channel = combined.getchannel("A")
            alpha_arr = np.array(alpha_channel)
            if np.count_nonzero(alpha_arr > 10) > 0:
                mask = Image.fromarray((alpha_arr > 10).astype(np.uint8) * 255, mode="L")
                return bg_img, mask

        return bg_img, None

    elif isinstance(editor_data, Image.Image):
        return editor_data.convert("RGB"), None

    return None, None


# --- PRESET SYNC HANDLERS ---
PRESET_DROPDOWN_CHOICES = [("None (Custom specification)", "none")] + [
    (v["label"], k) for k, v in FURNITURE_PRESETS.items()
]


def on_preset_select_batch(preset_key: str):
    if not preset_key or preset_key == "none" or preset_key not in FURNITURE_PRESETS:
        return gr.update(), "", "", "", gr.update(), DEFAULT_NEGATIVE_PROMPT

    p = FURNITURE_PRESETS[preset_key]
    part = p["part_text"]
    ch = p["change"]
    color = ch.get("color", "")
    material = ch.get("material", "")
    finish = ch.get("finish", "")
    pos, neg = build_prompt(part, ch)
    return part, color, material, finish, pos, neg


def on_structured_fields_change(part, color, material, finish, current_prompt):
    # Only auto-generate if structured attributes are supplied
    if not any([color, material, finish]):
        return current_prompt

    ch = {}
    if color and color.strip():
        ch["color"] = color.strip()
    if material and material.strip():
        ch["material"] = material.strip()
    if finish and finish.strip():
        ch["finish"] = finish.strip()

    pos, _ = build_prompt(part or "furniture component", ch)
    return pos


def on_preset_select_single(preset_key: str):
    if not preset_key or preset_key == "none" or preset_key not in FURNITURE_PRESETS:
        return gr.update(), gr.update(), DEFAULT_NEGATIVE_PROMPT

    p = FURNITURE_PRESETS[preset_key]
    part = p["part_text"]
    pos, neg = build_prompt(part, p["change"])
    return part, pos, neg


def on_generate_lstm(seed_text: str, temperature: float):
    """Generates specification line with CharLSTM, parses components, and auto-fills fields."""
    ckpt_path = Path("outputs/text/charlstm.pt")
    if not ckpt_path.exists():
        gr.Warning("No trained LSTM checkpoint found at outputs/text/charlstm.pt. Please train the model first using 'python -m src.text.train' or the Colab notebook.")
        return "", "chair legs", "", "", "", ""

    try:
        from src.text.generate import generate_from_checkpoint
        from src.text.spec_parser import parse_spec_line
        from src.prompts import build_prompt

        generated = generate_from_checkpoint(
            checkpoint_path=ckpt_path,
            seed=seed_text.strip() if seed_text else "Update: ",
            length=100,
            temperature=float(temperature),
            device="cpu",
            stop_at_newline=True,
        )

        parsed = parse_spec_line(generated)
        part = parsed["part"] or "chair legs"
        color = parsed["color"] or ""
        material = parsed["material"] or ""
        finish = parsed["finish"] or ""

        change = {"color": color, "material": material, "finish": finish}
        pos_prompt, _ = build_prompt(part, change)

        gr.Info(f"Generated spec for '{part}'!")
        return generated, part, color, material, finish, pos_prompt
    except Exception as e:
        gr.Warning(f"Text generation error: {e}")
        return str(e), "chair legs", "", "", "", ""


# --- TAB 1: BATCH PROCESSING HANDLER ---
def process_batch_ui(
    files,
    part_text: str,
    spec_prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    threshold: float,
    retry_larger_mask: bool,
    dilate_px: int,
    seed: int,
    use_controlnet: bool = DEFAULT_USE_CONTROLNET,
    controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
    progress=gr.Progress(track_tqdm=True),
):
    if not files:
        gr.Warning("Please upload at least one catalog image.")
        return [], [], [], [], None

    if not part_text or not part_text.strip():
        gr.Warning("Please specify the furniture part to change (e.g. 'chair legs').")
        return [], [], [], [], None

    if not spec_prompt or not spec_prompt.strip():
        gr.Warning("Please specify the new spec prompt (e.g. 'dark walnut wood legs').")
        return [], [], [], [], None

    temp_dir = tempfile.mkdtemp(prefix="catalog_batch_")
    in_dir = Path(temp_dir) / "input"
    out_dir = Path(temp_dir) / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        src_path = Path(f.name if hasattr(f, "name") else f)
        dest_path = in_dir / src_path.name
        with open(src_path, "rb") as rf, open(dest_path, "wb") as wf:
            wf.write(rf.read())

    def pcb(idx, total, name):
        progress((idx, total), desc=f"Processing {name} ({idx}/{total})")

    params = {
        "threshold": float(threshold),
        "retry_larger_mask": bool(retry_larger_mask),
        "retry_threshold": DEFAULT_RETRY_MASK_THRESHOLD,
        "dilate_px": int(dilate_px),
        "steps": int(steps),
        "guidance": float(guidance),
        "seed": int(seed),
        "use_controlnet": bool(use_controlnet),
        "controlnet_scale": float(controlnet_scale),
        "editor": editor,
    }

    manifest = run_batch(
        input_dir=in_dir,
        output_dir=out_dir,
        part_text=part_text.strip(),
        spec_prompt=spec_prompt.strip(),
        negative_prompt=negative_prompt.strip() if negative_prompt else None,
        params=params,
        progress_cb=pcb,
    )

    zip_path = zip_outputs(out_dir)

    before_after_gallery = []
    masks_gallery = []
    controls_gallery = []
    table_rows = []

    success_count = 0
    for item in manifest:
        table_rows.append([
            item["name"],
            item["status"],
            f"{item['mask_coverage_pct']}%",
            f"{item['time_s']}s",
            item["output_path"] if item["output_path"] else "Skipped",
        ])

        mask_file = out_dir / "masks" / f"{Path(item['name']).stem}_mask.png"
        if mask_file.exists():
            masks_gallery.append((str(mask_file), f"Mask: {item['name']} ({item['mask_coverage_pct']}%)"))

        ctrl_file = out_dir / "control" / f"{Path(item['name']).stem}_control.png"
        if ctrl_file.exists():
            controls_gallery.append((str(ctrl_file), f"Edge Map: {item['name']}"))

        if "success" in item["status"] and item["output_path"]:
            success_count += 1
            orig_file = in_dir / item["name"]
            before_after_gallery.append((str(orig_file), f"Before: {item['name']}"))
            before_after_gallery.append((item["output_path"], f"After: {spec_prompt[:35]}..."))

    if success_count == 0:
        gr.Warning(f"No parts matching '{part_text}' were found (coverage < 0.5%). Check part description or threshold.")
    else:
        gr.Info(f"Batch completed: {success_count}/{len(manifest)} images successfully updated!")

    return before_after_gallery, masks_gallery, controls_gallery, table_rows, zip_path


# --- TAB 2: SINGLE IMAGE HANDLERS ---
def generate_auto_mask_ui(editor_data, part_text: str, threshold: float, dilate_px: int):
    bg_img, _ = extract_mask_from_editor(editor_data)
    if bg_img is None:
        gr.Warning("Please upload an image into the editor first.")
        return None, None

    if not part_text or not part_text.strip():
        gr.Warning("Please specify the furniture part to detect (e.g. 'chair legs').")
        return None, None

    mask = editor.generate_mask(
        image=bg_img,
        part_text=part_text.strip(),
        threshold=float(threshold),
        dilate_px=int(dilate_px),
    )

    mask_arr = np.array(mask)
    coverage = round((np.count_nonzero(mask_arr > 0) / mask_arr.size) * 100.0, 2)

    overlay_arr = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    overlay_arr[mask_arr > 0] = [255, 30, 30, 160]
    overlay_layer = Image.fromarray(overlay_arr, mode="RGBA")

    updated_editor = {
        "background": bg_img,
        "layers": [overlay_layer],
        "composite": bg_img,
    }

    if coverage < 0.5:
        gr.Warning(f"Part '{part_text}' was under 0.5% coverage ({coverage}%). You can manually brush the mask.")
    else:
        gr.Info(f"Detected '{part_text}' with {coverage}% coverage.")

    return updated_editor, mask


def process_single_ui(
    editor_data,
    mask_state,
    part_text: str,
    spec_prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    threshold: float,
    dilate_px: int,
    seed: int,
    use_controlnet: bool = DEFAULT_USE_CONTROLNET,
    controlnet_scale: float = DEFAULT_CONTROLNET_SCALE,
    progress=gr.Progress(),
):
    bg_img, drawn_mask = extract_mask_from_editor(editor_data)
    if bg_img is None:
        gr.Warning("Please upload an image into the editor first.")
        return None, None, None, None

    if not spec_prompt or not spec_prompt.strip():
        gr.Warning("Please specify the new spec prompt (e.g. 'dark walnut wood').")
        return None, None, None, None

    if drawn_mask is not None:
        final_mask = drawn_mask
    elif mask_state is not None:
        final_mask = mask_state.convert("L")
    else:
        if not part_text or not part_text.strip():
            gr.Warning("Please enter a part description or paint a mask.")
            return None, None, None, None
        final_mask = editor.generate_mask(bg_img, part_text.strip(), threshold=float(threshold), dilate_px=int(dilate_px))

    mask_arr = np.array(final_mask)
    coverage = round((np.count_nonzero(mask_arr > 0) / mask_arr.size) * 100.0, 2)
    if coverage < 0.5:
        gr.Warning(f"Mask is empty or under 0.5% coverage ({coverage}%). Paint a mask or lower threshold.")
        control_preview = editor.make_control_image(bg_img)
        return bg_img, None, final_mask, control_preview

    progress((0, 1), desc="Inpainting new specification...")
    result = editor.inpaint(
        image=bg_img,
        mask=final_mask,
        prompt=spec_prompt.strip(),
        negative_prompt=negative_prompt.strip() if negative_prompt else "",
        steps=int(steps),
        guidance=float(guidance),
        seed=int(seed),
        use_controlnet=bool(use_controlnet),
        controlnet_scale=float(controlnet_scale),
    )
    control_img = editor.make_control_image(bg_img)
    gr.Info("Inpainting finished successfully!")
    return bg_img, result, final_mask, control_img


# --- BUILD GRADIO INTERFACE ---
with gr.Blocks(title="CatalogUpdate AI - Furniture Catalog Editor") as demo:
    gr.Markdown(
        """
        # CatalogUpdate AI
        ### Intelligent Batch Product-Image Editor for Furniture Catalogs
        Zero-shot part segmentation with CLIPSeg and specification inpainting with Stable Diffusion (v1.5 FP16).
        """
    )

    with gr.Tabs():
        # --- TAB 1: BATCH UPDATE ---
        with gr.Tab("Batch Catalog Update"):
            with gr.Row():
                with gr.Column(scale=4):
                    batch_files = gr.File(
                        label="Upload Catalog Images (PNG, JPG, WEBP)",
                        file_count="multiple",
                        file_types=["image"],
                    )

                    batch_preset = gr.Dropdown(
                        choices=PRESET_DROPDOWN_CHOICES,
                        value="none",
                        label="Quick Furniture Preset (Auto-fills spec attributes)",
                    )

                    with gr.Accordion("Generate a spec update (LSTM)", open=False):
                        with gr.Row():
                            lstm_seed = gr.Textbox(
                                label="Seed text prompt",
                                value="Update: chair legs changed to",
                                placeholder="e.g. Update: chair legs changed to",
                                scale=3,
                            )
                            lstm_temp = gr.Slider(
                                0.2, 1.5,
                                value=0.8,
                                step=0.05,
                                label="Temperature",
                                scale=1,
                            )
                        lstm_gen_btn = gr.Button("Generate Spec with LSTM", variant="secondary")
                        lstm_out = gr.Textbox(
                            label="Generated Spec String (auto-populates fields below)",
                            interactive=False,
                            lines=1,
                        )

                    batch_part = gr.Textbox(
                        label="Part to change",
                        value="chair legs",
                        placeholder="e.g. chair legs, cushion, tabletop",
                    )

                    with gr.Row():
                        batch_color = gr.Textbox(label="Color", placeholder="e.g. dark walnut")
                        batch_material = gr.Textbox(label="Material", placeholder="e.g. solid walnut wood")
                        batch_finish = gr.Textbox(label="Finish", placeholder="e.g. natural satin")

                    batch_prompt = gr.Textbox(
                        label="New spec prompt (Structured or free-text)",
                        value="chair legs made of solid walnut wood, dark walnut, natural satin finish, studio product photo, white background, sharp, high detail, consistent lighting",
                        lines=2,
                    )

                    with gr.Accordion("Advanced Inpainting & Masking Parameters", open=False):
                        with gr.Row():
                            batch_steps = gr.Slider(10, 50, value=DEFAULT_STEPS, step=1, label="Inference Steps")
                            batch_guidance = gr.Slider(1.0, 15.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale")
                        with gr.Row():
                            batch_threshold = gr.Slider(0.1, 0.9, value=DEFAULT_MASK_THRESHOLD, step=0.05, label="Mask Threshold")
                            batch_dilate = gr.Slider(0, 30, value=DEFAULT_MASK_DILATION_PX, step=1, label="Mask Dilation (px)")
                            batch_seed = gr.Number(value=DEFAULT_SEED, label="Random Seed", precision=0)
                        with gr.Row():
                            batch_use_controlnet = gr.Checkbox(
                                label="Preserve structure (ControlNet)",
                                value=DEFAULT_USE_CONTROLNET,
                            )
                            batch_controlnet_scale = gr.Slider(
                                0.0, 1.5,
                                value=DEFAULT_CONTROLNET_SCALE,
                                step=0.05,
                                label="ControlNet strength",
                            )
                        batch_retry_mask = gr.Checkbox(
                            label="Retry with larger mask if coverage < 0.5%",
                            value=True,
                        )
                        batch_neg = gr.Textbox(
                            label="Negative Prompt",
                            value=DEFAULT_NEGATIVE_PROMPT,
                        )

                    batch_run_btn = gr.Button("Run Batch Update", variant="primary", size="lg")

                with gr.Column(scale=5):
                    batch_zip_out = gr.File(label="Download Processed ZIP Package")
                    batch_gallery = gr.Gallery(
                        label="Before / After Comparisons",
                        columns=2,
                        height="auto",
                    )
                    batch_masks = gr.Gallery(
                        label="Generated Component Masks",
                        columns=3,
                        height="auto",
                    )
                    batch_controls = gr.Gallery(
                        label="ControlNet Edge Maps (Canny)",
                        columns=3,
                        height="auto",
                    )
                    batch_table = gr.Dataframe(
                        headers=["Name", "Status", "Mask Coverage (%)", "Time (s)", "Output Path"],
                        label="Results Manifest",
                    )

            # Wiring Preset Selection in Batch
            batch_preset.change(
                fn=on_preset_select_batch,
                inputs=[batch_preset],
                outputs=[batch_part, batch_color, batch_material, batch_finish, batch_prompt, batch_neg],
            )

            # Wiring LSTM Spec Generation
            lstm_gen_btn.click(
                fn=on_generate_lstm,
                inputs=[lstm_seed, lstm_temp],
                outputs=[lstm_out, batch_part, batch_color, batch_material, batch_finish, batch_prompt],
            )

            # Wiring Structured Spec inputs to auto-update prompt
            for inp in [batch_color, batch_material, batch_finish]:
                inp.blur(
                    fn=on_structured_fields_change,
                    inputs=[batch_part, batch_color, batch_material, batch_finish, batch_prompt],
                    outputs=[batch_prompt],
                )

            batch_run_btn.click(
                fn=process_batch_ui,
                inputs=[
                    batch_files,
                    batch_part,
                    batch_prompt,
                    batch_neg,
                    batch_steps,
                    batch_guidance,
                    batch_threshold,
                    batch_retry_mask,
                    batch_dilate,
                    batch_seed,
                    batch_use_controlnet,
                    batch_controlnet_scale,
                ],
                outputs=[batch_gallery, batch_masks, batch_controls, batch_table, batch_zip_out],
            )

        # --- TAB 2: SINGLE IMAGE INTERACTIVE ---
        with gr.Tab("Single Image (Manual Mask)"):
            single_mask_state = gr.State()

            with gr.Row():
                with gr.Column(scale=5):
                    single_editor = gr.ImageEditor(
                        label="Product Image (Brush to paint manual mask, or use Auto-mask)",
                        type="pil",
                        brush=gr.Brush(colors=["#FF2222"], default_size=25),
                        sources=["upload", "clipboard"],
                    )

                    single_preset = gr.Dropdown(
                        choices=PRESET_DROPDOWN_CHOICES,
                        value="none",
                        label="Quick Furniture Preset",
                    )

                    with gr.Row():
                        single_part = gr.Textbox(label="Part to change", value="chair legs")
                        single_auto_mask_btn = gr.Button("Auto-mask with CLIPSeg")

                    single_prompt = gr.Textbox(
                        label="New spec",
                        value="chair legs made of solid walnut wood, dark walnut, natural satin finish, studio product photo, white background, sharp, high detail, consistent lighting",
                        lines=2,
                    )

                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row():
                            single_steps = gr.Slider(10, 50, value=DEFAULT_STEPS, step=1, label="Inference Steps")
                            single_guidance = gr.Slider(1.0, 15.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale")
                        with gr.Row():
                            single_threshold = gr.Slider(0.1, 0.9, value=DEFAULT_MASK_THRESHOLD, step=0.05, label="Mask Threshold")
                            single_dilate = gr.Slider(0, 30, value=DEFAULT_MASK_DILATION_PX, step=1, label="Mask Dilation (px)")
                            single_seed = gr.Number(value=DEFAULT_SEED, label="Random Seed", precision=0)
                        with gr.Row():
                            single_use_controlnet = gr.Checkbox(
                                label="Preserve structure (ControlNet)",
                                value=DEFAULT_USE_CONTROLNET,
                            )
                            single_controlnet_scale = gr.Slider(
                                0.0, 1.5,
                                value=DEFAULT_CONTROLNET_SCALE,
                                step=0.05,
                                label="ControlNet strength",
                            )
                        single_neg = gr.Textbox(
                            label="Negative Prompt",
                            value=DEFAULT_NEGATIVE_PROMPT,
                        )

                    single_run_btn = gr.Button("Inpaint New Spec", variant="primary", size="lg")

                with gr.Column(scale=5):
                    with gr.Row():
                        single_orig_view = gr.Image(label="Original Image", type="pil")
                        single_mask_view = gr.Image(label="Mask Used (White = Edit)", type="pil")
                        single_control_view = gr.Image(label="ControlNet Edge Map", type="pil")
                    single_result_view = gr.Image(label="Inpainted Spec Result", type="pil")

            single_preset.change(
                fn=on_preset_select_single,
                inputs=[single_preset],
                outputs=[single_part, single_prompt, single_neg],
            )

            # Connect Auto-mask button
            single_auto_mask_btn.click(
                fn=generate_auto_mask_ui,
                inputs=[single_editor, single_part, single_threshold, single_dilate],
                outputs=[single_editor, single_mask_state],
            )

            # Connect Inpaint button
            single_run_btn.click(
                fn=process_single_ui,
                inputs=[
                    single_editor,
                    single_mask_state,
                    single_part,
                    single_prompt,
                    single_neg,
                    single_steps,
                    single_guidance,
                    single_threshold,
                    single_dilate,
                    single_seed,
                    single_use_controlnet,
                    single_controlnet_scale,
                ],
                outputs=[single_orig_view, single_result_view, single_mask_view, single_control_view],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatalogUpdate AI App")
    parser.add_argument("--share", action="store_true", help="Launch with public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--mock", action="store_true", help="Use mock engine on CPU")
    parsed_args, _ = parser.parse_known_args()

    should_share = parsed_args.share or is_colab()
    print(f"Starting CatalogUpdate AI Gradio App (share={should_share})...")
    demo.queue().launch(
        share=should_share,
        server_name="0.0.0.0",
        server_port=parsed_args.port,
    )
