"""
CatalogUpdate AI - Spec-Aware Prompt Engineering & Furniture Presets.

Generates studio-grade product catalog prompts from structured specification changes
(material, color, finish, shape/style, extra details) and provides curated furniture presets.
"""

from typing import Dict, Any, Tuple

# Shared negative prompt preventing typical diffusion artifacts and background bleed
DEFAULT_NEGATIVE_PROMPT: str = (
    "blurry, distorted, deformed, extra legs, extra parts, text, watermark, "
    "low quality, different background, floating, artifacts, seams, color bleeding"
)

# 12 Curated Furniture Presets
FURNITURE_PRESETS: Dict[str, Dict[str, Any]] = {
    "chair_walnut_legs": {
        "label": "Chair legs -> Walnut wood",
        "part_text": "chair legs",
        "change": {
            "color": "dark walnut",
            "material": "solid walnut wood",
            "finish": "natural satin",
            "shape/style": "clean tapered",
            "extra": "rich natural wood grain",
        },
    },
    "sofa_charcoal_linen": {
        "label": "Sofa fabric -> Charcoal grey linen",
        "part_text": "sofa fabric upholstery",
        "change": {
            "color": "charcoal grey",
            "material": "textured woven linen",
            "finish": "matte soft",
            "shape/style": "contemporary tailored",
            "extra": "fine fabric weave texture",
        },
    },
    "table_carrara_marble": {
        "label": "Table top -> White Carrara marble",
        "part_text": "tabletop surface",
        "change": {
            "color": "white",
            "material": "natural Carrara marble",
            "finish": "polished honed",
            "shape/style": "smooth slab edge",
            "extra": "delicate grey stone veining",
        },
    },
    "cabinet_brushed_brass": {
        "label": "Cabinet handles -> Brushed brass",
        "part_text": "cabinet door handles and drawer pulls",
        "change": {
            "color": "warm gold",
            "material": "solid brass",
            "finish": "brushed metallic",
            "shape/style": "minimalist linear bar",
            "extra": "subtle metallic sheen",
        },
    },
    "armrests_black_leather": {
        "label": "Armrests -> Black Italian leather",
        "part_text": "chair armrests",
        "change": {
            "color": "deep black",
            "material": "full-grain Italian leather",
            "finish": "semi-gloss supple",
            "shape/style": "taut upholstered",
            "extra": "subtle natural pebble grain",
        },
    },
    "cushion_teal_velvet": {
        "label": "Cushion -> Teal velvet",
        "part_text": "seat cushion",
        "change": {
            "color": "rich jewel teal",
            "material": "plush velvet fabric",
            "finish": "soft sheen",
            "shape/style": "neat piping border",
            "extra": "luxurious tactile pile",
        },
    },
    "wood_matte_oak": {
        "label": "Wood finish -> Matte natural oak",
        "part_text": "wooden frame",
        "change": {
            "color": "light natural oak",
            "material": "white oak wood",
            "finish": "matte protective clear coat",
            "shape/style": "Nordic minimalist",
            "extra": "straight organic wood grain",
        },
    },
    "chair_beige_boucle": {
        "label": "Chair seat -> Beige boucle fabric",
        "part_text": "chair seat upholstery",
        "change": {
            "color": "warm beige cream",
            "material": "heavy boucle wool fabric",
            "finish": "matte nubby",
            "shape/style": "curved rounded form",
            "extra": "rich looped textured yarn",
        },
    },
    "desk_matte_black_steel": {
        "label": "Desk frame -> Matte black steel",
        "part_text": "desk legs and metal frame",
        "change": {
            "color": "industrial matte black",
            "material": "powder-coated steel",
            "finish": "fine texture matte",
            "shape/style": "geometric square tubing",
            "extra": "seamless welded joints",
        },
    },
    "dining_woven_rattan": {
        "label": "Chair backrest -> Woven rattan cane",
        "part_text": "chair backrest",
        "change": {
            "color": "natural honey cane",
            "material": "woven rattan webbing",
            "finish": "natural semi-matte",
            "shape/style": "octagonal open weave",
            "extra": "crisp handwoven lattice pattern",
        },
    },
    "table_chrome_base": {
        "label": "Coffee table base -> Polished chrome",
        "part_text": "table base legs",
        "change": {
            "color": "mirror silver",
            "material": "chrome plated steel",
            "finish": "high gloss mirror polished",
            "shape/style": "sculptural curved tubular",
            "extra": "crisp specular studio reflections",
        },
    },
    "headboard_emerald_velvet": {
        "label": "Headboard -> Emerald green velvet",
        "part_text": "bed headboard panel",
        "change": {
            "color": "deep emerald green",
            "material": "cotton velvet upholstery",
            "finish": "soft lustrous sheen",
            "shape/style": "vertical channel tufting",
            "extra": "luxurious depth and soft drape",
        },
    },
}


def build_prompt(part: str, change: Dict[str, Any]) -> Tuple[str, str]:
    """
    Composes a commercial product-photography prompt from part and specification attributes.

    Args:
        part: Target component name (e.g. 'chair legs').
        change: Dict with keys: color, material, finish, shape/style, extra.

    Returns:
        Tuple of (positive_prompt, negative_prompt).
    """
    elements = []

    part_str = part.strip() if part else "furniture component"
    material = str(change.get("material", "")).strip()
    color = str(change.get("color", "")).strip()
    finish = str(change.get("finish", "")).strip()
    style = str(change.get("shape/style", "")).strip()
    extra = str(change.get("extra", "")).strip()

    # Core component and material description
    if material:
        elements.append(f"{part_str} made of {material}")
    else:
        elements.append(part_str)

    if color:
        elements.append(color)

    if finish:
        if "finish" in finish.lower():
            elements.append(finish)
        else:
            elements.append(f"{finish} finish")

    if style:
        elements.append(style)

    if extra:
        elements.append(extra)

    # Standard commercial furniture catalog studio quality keywords
    elements.extend([
        "studio product photo",
        "white background",
        "sharp",
        "high detail",
        "consistent lighting"
    ])

    positive_prompt = ", ".join(elements)
    negative_prompt = DEFAULT_NEGATIVE_PROMPT

    return positive_prompt, negative_prompt


def main():
    """Demo CLI printing generated prompts for all 12 furniture presets."""
    print("=" * 80)
    print("CatalogUpdate AI - Spec-Aware Prompt Presets Demonstration")
    print("=" * 80)
    for key, data in FURNITURE_PRESETS.items():
        pos, neg = build_prompt(data["part_text"], data["change"])
        print(f"\n[PRESET] {data['label']} (Key: {key})")
        print(f"  Target Part     : {data['part_text']}")
        print(f"  Positive Prompt : {pos}")
    print("\n" + "-" * 80)
    print("Shared Negative Prompt:")
    print(f"  {DEFAULT_NEGATIVE_PROMPT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
