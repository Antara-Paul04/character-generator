"""
Hair generation module using GBH Tool
Maps text prompts to GBH Tool parameters for procedural hair generation
Now includes programmatic sample generation for diverse hair output
"""

import re
import random
import itertools
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


# =============================================================================
# HAIR SAMPLE DEFINITIONS
# =============================================================================

# All named archetypes with their canonical parameter sets
HAIR_ARCHETYPES = {
    # ── Straight styles ──────────────────────────────────────────────────────
    "straight_sleek": {
        "style": "straight", "length": "long", "volume": "thin",
        "conversion_type": "CURVES",
        "params": {"curl": 0.0, "noise": 0.1, "density": 0.65, "length": 0.75,
                   "parting": 0.5, "shine": 0.9, "thickness": 0.3}
    },
    "straight_thick": {
        "style": "straight", "length": "medium", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"curl": 0.0, "noise": 0.15, "density": 0.85, "length": 0.5,
                   "parting": 0.4, "shine": 0.7, "thickness": 0.6}
    },
    "straight_short": {
        "style": "short", "length": "short", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"curl": 0.0, "noise": 0.1, "density": 0.7, "length": 0.25,
                   "parting": 0.5, "shine": 0.75, "thickness": 0.45}
    },

    # ── Wavy styles ──────────────────────────────────────────────────────────
    "wavy_beach": {
        "style": "wavy", "length": "long", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"curl": 0.35, "noise": 0.3, "density": 0.7, "length": 0.72,
                   "parting": 0.45, "shine": 0.6, "thickness": 0.45}
    },
    "wavy_medium": {
        "style": "wavy", "length": "medium", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"curl": 0.42, "noise": 0.28, "density": 0.8, "length": 0.5,
                   "parting": 0.5, "shine": 0.55, "thickness": 0.5}
    },
    "wavy_loose": {
        "style": "wavy", "length": "very_long", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"curl": 0.25, "noise": 0.35, "density": 0.75, "length": 0.9,
                   "parting": 0.48, "shine": 0.62, "thickness": 0.52}
    },

    # ── Curly styles ─────────────────────────────────────────────────────────
    "curly_tight": {
        "style": "curly", "length": "medium", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"curl": 0.85, "noise": 0.4, "density": 0.88, "length": 0.48,
                   "parting": 0.5, "shine": 0.4, "thickness": 0.55}
    },
    "curly_loose": {
        "style": "curly", "length": "long", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"curl": 0.6, "noise": 0.35, "density": 0.82, "length": 0.65,
                   "parting": 0.5, "shine": 0.45, "thickness": 0.5}
    },
    "curly_short": {
        "style": "curly", "length": "short", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"curl": 0.78, "noise": 0.38, "density": 0.8, "length": 0.28,
                   "parting": 0.5, "shine": 0.42, "thickness": 0.48}
    },
    "ringlets": {
        "style": "curly", "length": "long", "volume": "very_thick",
        "conversion_type": "CURVES",
        "params": {"curl": 0.95, "noise": 0.45, "density": 0.92, "length": 0.7,
                   "parting": 0.5, "shine": 0.35, "thickness": 0.6}
    },

    # ── Kinky / Afro styles ──────────────────────────────────────────────────
    "kinky_natural": {
        "style": "kinky", "length": "short", "volume": "very_thick",
        "conversion_type": "CURVES",
        "params": {"curl": 1.0, "noise": 0.55, "density": 0.95, "length": 0.25,
                   "volume": 0.9, "parting": 0.5, "shine": 0.25, "thickness": 0.65}
    },
    "afro_full": {
        "style": "afro", "length": "medium", "volume": "very_thick",
        "conversion_type": "CURVES",
        "params": {"curl": 1.0, "noise": 0.6, "density": 1.0, "length": 0.45,
                   "volume": 1.0, "parting": 0.5, "shine": 0.2, "thickness": 0.7}
    },
    "afro_large": {
        "style": "afro", "length": "long", "volume": "very_thick",
        "conversion_type": "CURVES",
        "params": {"curl": 1.0, "noise": 0.65, "density": 1.0, "length": 0.6,
                   "volume": 1.0, "parting": 0.5, "shine": 0.18, "thickness": 0.75}
    },

    # ── Protective / Textured styles ─────────────────────────────────────────
    "braided_classic": {
        "style": "braided", "length": "long", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"braid_intensity": 0.8, "twist": 1.0, "density": 0.78,
                   "length": 0.72, "curl": 0.0, "thickness": 0.55}
    },
    "cornrows": {
        "style": "cornrows", "length": "short", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"braid_intensity": 1.0, "pattern": 0.5, "density": 0.72,
                   "length": 0.2, "curl": 0.0, "thickness": 0.4}
    },
    "dreadlocks_short": {
        "style": "dreadlocks", "length": "short", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"twist": 1.0, "curl": 0.15, "density": 0.6,
                   "length": 0.3, "thickness": 0.8}
    },
    "dreadlocks_long": {
        "style": "dreadlocks", "length": "long", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"twist": 1.0, "curl": 0.2, "density": 0.62,
                   "length": 0.75, "thickness": 0.85}
    },
    "twisted_out": {
        "style": "twisted", "length": "medium", "volume": "thick",
        "conversion_type": "CURVES",
        "params": {"twist": 0.8, "curl": 0.35, "density": 0.72,
                   "length": 0.48, "shine": 0.3, "thickness": 0.58}
    },

    # ── Short / Cropped styles ───────────────────────────────────────────────
    "pixie_cut": {
        "style": "pixie", "length": "very_short", "volume": "thin",
        "conversion_type": "CURVES",
        "params": {"curl": 0.0, "noise": 0.12, "density": 0.68, "length": 0.1,
                   "style_type": 0.3, "shine": 0.7, "thickness": 0.35}
    },
    "buzz_cut": {
        "style": "short", "length": "very_short", "volume": "thin",
        "conversion_type": "CURVES",
        "params": {"curl": 0.0, "noise": 0.05, "density": 0.6, "length": 0.05,
                   "shine": 0.8, "thickness": 0.2}
    },
    "undercut_style": {
        "style": "undercut", "length": "medium", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"sides": 0.0, "top": 0.7, "style_type": 5.0,
                   "curl": 0.0, "density": 0.72, "length": 0.45, "thickness": 0.4}
    },
    "mohawk_style": {
        "style": "mohawk", "length": "medium", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"mohawk_width": 0.7, "style_type": 3.0, "length": 0.4,
                   "curl": 0.0, "density": 0.65, "spike_height": 0.6}
    },
    "quiff_style": {
        "style": "quiff", "length": "medium", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"front_volume": 0.85, "length": 0.48, "style_type": 6.0,
                   "curl": 0.1, "density": 0.7, "shine": 0.65}
    },
    "spiky_style": {
        "style": "spiky", "length": "short", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"spike_height": 0.72, "spike_density": 0.62, "style_type": 7.0,
                   "curl": 0.0, "density": 0.65, "length": 0.25}
    },

    # ── Updo / Styled styles ─────────────────────────────────────────────────
    "bun_top": {
        "style": "bun", "length": "short", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"bun_size": 0.72, "style_type": 2.0, "length": 0.18,
                   "curl": 0.05, "density": 0.7, "shine": 0.6}
    },
    "ponytail_high": {
        "style": "ponytail", "length": "medium", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"ponytail_height": 0.82, "style_type": 1.0, "length": 0.55,
                   "curl": 0.05, "density": 0.72, "shine": 0.65}
    },
    "ponytail_low": {
        "style": "ponytail", "length": "long", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"ponytail_height": 0.25, "style_type": 1.0, "length": 0.68,
                   "curl": 0.1, "density": 0.72, "shine": 0.6}
    },

    # ── Fantasy / Special styles ─────────────────────────────────────────────
    "anime_long": {
        "style": "straight", "length": "very_long", "volume": "thick",
        "conversion_type": "MESH",
        "params": {"curl": 0.0, "noise": 0.08, "density": 0.9, "length": 0.92,
                   "shine": 0.95, "thickness": 0.5, "parting": 0.5}
    },
    "anime_spiky": {
        "style": "spiky", "length": "medium", "volume": "thick",
        "conversion_type": "MESH",
        "params": {"spike_height": 0.85, "spike_density": 0.75, "style_type": 7.0,
                   "curl": 0.0, "density": 0.85, "length": 0.42, "shine": 0.8}
    },
    "fantasy_elven": {
        "style": "straight", "length": "very_long", "volume": "thin",
        "conversion_type": "CURVES",
        "params": {"curl": 0.05, "noise": 0.08, "density": 0.7, "length": 0.95,
                   "shine": 0.92, "thickness": 0.28, "parting": 0.5}
    },
    "mullet_classic": {
        "style": "mullet", "length": "medium", "volume": "normal",
        "conversion_type": "CURVES",
        "params": {"length_front": 0.2, "length_back": 0.8, "style_type": 4.0,
                   "curl": 0.0, "density": 0.68, "length": 0.55}
    },
}

# Color presets (R, G, B, A) for use in GBH material nodes
HAIR_COLOR_PRESETS = {
    "black":          (0.02, 0.02, 0.02, 1.0),
    "dark_brown":     (0.08, 0.04, 0.02, 1.0),
    "brown":          (0.18, 0.09, 0.04, 1.0),
    "light_brown":    (0.32, 0.18, 0.08, 1.0),
    "auburn":         (0.28, 0.10, 0.04, 1.0),
    "dark_blonde":    (0.45, 0.30, 0.12, 1.0),
    "blonde":         (0.72, 0.55, 0.22, 1.0),
    "platinum":       (0.92, 0.88, 0.78, 1.0),
    "ash_blonde":     (0.75, 0.72, 0.64, 1.0),
    "strawberry":     (0.65, 0.38, 0.22, 1.0),
    "red":            (0.45, 0.08, 0.04, 1.0),
    "ginger":         (0.52, 0.20, 0.06, 1.0),
    "gray":           (0.55, 0.55, 0.55, 1.0),
    "silver":         (0.78, 0.78, 0.80, 1.0),
    "white":          (0.95, 0.95, 0.95, 1.0),
    "salt_pepper":    (0.35, 0.35, 0.35, 1.0),
    "blue":           (0.10, 0.20, 0.75, 1.0),
    "pink":           (0.85, 0.35, 0.55, 1.0),
    "purple":         (0.40, 0.10, 0.70, 1.0),
    "green":          (0.10, 0.55, 0.20, 1.0),
    "teal":           (0.08, 0.55, 0.52, 1.0),
}

# Keyword → color name mapping (matches text prompts)
COLOR_KEYWORD_MAP = {
    "black": "black", "jet black": "black", "raven": "black",
    "dark brown": "dark_brown", "chocolate": "dark_brown",
    "brown": "brown", "brunette": "brown", "chestnut": "brown",
    "light brown": "light_brown", "caramel": "light_brown",
    "auburn": "auburn", "copper": "auburn",
    "dark blonde": "dark_blonde", "dirty blonde": "dark_blonde",
    "blonde": "blonde", "blond": "blonde", "golden": "blonde",
    "platinum": "platinum", "platinum blonde": "platinum",
    "ash blonde": "ash_blonde", "ash": "ash_blonde",
    "strawberry blonde": "strawberry", "strawberry": "strawberry",
    "red": "red", "fiery red": "red",
    "ginger": "ginger", "orange": "ginger",
    "gray": "gray", "grey": "gray",
    "silver": "silver",
    "white": "white",
    "salt and pepper": "salt_pepper",
    "blue": "blue", "cyan": "blue",
    "pink": "pink", "rose": "pink",
    "purple": "purple", "lavender": "purple", "violet": "purple",
    "green": "green", "emerald": "green",
    "teal": "teal",
}


# =============================================================================
# PROGRAMMATIC SAMPLE GENERATOR
# =============================================================================

class HairSampleGenerator:
    """
    Generates a large, diverse set of hair parameter samples programmatically.

    Three generation strategies are combined:
      1. Archetype samples  – the curated presets above, verbatim
      2. Variation samples  – each archetype ± random jitter on every numeric param
      3. Grid samples       – systematic sweep across key axes (style × length × volume)
    """

    # Axes used for grid sampling
    STYLES   = ["straight", "wavy", "curly", "kinky", "braided", "dreadlocks",
                "pixie", "bun", "ponytail", "afro", "undercut", "spiky", "quiff"]
    LENGTHS  = ["very_short", "short", "medium", "long", "very_long"]
    VOLUMES  = ["thin", "normal", "thick", "very_thick"]
    CONV_TYPES = ["CURVES", "MESH"]

    LENGTH_VALUES = {
        "very_short": 0.08, "short": 0.25, "medium": 0.50,
        "long": 0.72, "very_long": 0.92
    }
    VOLUME_DENSITY = {
        "thin": 0.55, "normal": 0.70, "thick": 0.85, "very_thick": 0.97
    }

    # Numeric parameters that are safe to jitter
    JITTER_PARAMS = [
        "curl", "noise", "density", "length", "volume", "shine", "thickness",
        "front_volume", "bangs", "bangs_length", "spike_height", "spike_density",
        "bun_size", "ponytail_height", "mohawk_width", "braid_intensity", "twist",
        "pattern", "sides", "top"
    ]
    JITTER_RANGE = 0.12   # ± this amount, clamped to [0, 1]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._archetype_list = list(HAIR_ARCHETYPES.keys())

    # ── Public API ────────────────────────────────────────────────────────────

    def get_archetype_samples(self) -> List[Dict]:
        """Return the curated archetype library as sample dicts."""
        samples = []
        for name, arch in HAIR_ARCHETYPES.items():
            sample = self._build_sample(
                name=name,
                style=arch["style"],
                length=arch["length"],
                volume=arch.get("volume", "normal"),
                conversion_type=arch.get("conversion_type", "CURVES"),
                params=dict(arch["params"]),
                source="archetype"
            )
            samples.append(sample)
        return samples

    def get_variation_samples(self, variations_per_archetype: int = 4) -> List[Dict]:
        """
        For each archetype, create N jittered variants.
        Produces len(ARCHETYPES) * variations_per_archetype samples.
        """
        samples = []
        for name, arch in HAIR_ARCHETYPES.items():
            for v in range(variations_per_archetype):
                jittered = self._jitter_params(dict(arch["params"]))
                sample = self._build_sample(
                    name=f"{name}_var{v+1}",
                    style=arch["style"],
                    length=arch["length"],
                    volume=arch.get("volume", "normal"),
                    conversion_type=arch.get("conversion_type", "CURVES"),
                    params=jittered,
                    source="variation"
                )
                samples.append(sample)
        return samples

    def get_grid_samples(self) -> List[Dict]:
        """
        Systematic sweep: every (style × length) combination with
        two volume tiers and a deterministic parameter set.
        """
        samples = []
        for style in self.STYLES:
            for length in self.LENGTHS:
                for volume in ["normal", "thick"]:      # two tiers keeps it manageable
                    name = f"grid_{style}_{length}_{volume}"
                    params = self._grid_params(style, length, volume)
                    sample = self._build_sample(
                        name=name,
                        style=style,
                        length=length,
                        volume=volume,
                        conversion_type="CURVES",
                        params=params,
                        source="grid"
                    )
                    samples.append(sample)
        return samples

    def get_color_samples(self) -> List[Dict]:
        """
        One sample per color preset paired with common styles.
        """
        paired_styles = [
            ("straight", "long",   "thin"),
            ("wavy",     "medium", "normal"),
            ("curly",    "medium", "thick"),
            ("afro",     "medium", "very_thick"),
            ("pixie",    "very_short", "thin"),
        ]
        samples = []
        for color_name, color_rgba in HAIR_COLOR_PRESETS.items():
            style, length, volume = random.choice(paired_styles)
            params = self._grid_params(style, length, volume)
            params["color_r"] = color_rgba[0]
            params["color_g"] = color_rgba[1]
            params["color_b"] = color_rgba[2]
            params["color_a"] = color_rgba[3]

            sample = self._build_sample(
                name=f"color_{color_name}_{style}",
                style=style,
                length=length,
                volume=volume,
                conversion_type="CURVES",
                params=params,
                source="color"
            )
            sample["color"] = color_name
            sample["color_rgba"] = color_rgba
            samples.append(sample)
        return samples

    def get_all_samples(
        self,
        variations_per_archetype: int = 4,
        include_grid: bool = True,
        include_colors: bool = True
    ) -> List[Dict]:
        """
        Combine all generation strategies into one deduplicated list.
        """
        all_samples = []
        all_samples += self.get_archetype_samples()
        all_samples += self.get_variation_samples(variations_per_archetype)
        if include_grid:
            all_samples += self.get_grid_samples()
        if include_colors:
            all_samples += self.get_color_samples()

        # Deduplicate by name (keep first occurrence)
        seen = set()
        unique = []
        for s in all_samples:
            if s["name"] not in seen:
                seen.add(s["name"])
                unique.append(s)

        return unique

    def get_samples_for_style(self, style: str, count: int = 8) -> List[Dict]:
        """
        Return `count` samples that match a given style keyword.
        Pulls from archetypes first, then fills with variations / grid.
        """
        all_samples = self.get_all_samples()
        matched = [s for s in all_samples if s.get("style") == style]

        if len(matched) >= count:
            return matched[:count]

        # Not enough – generate extras on the fly
        extras = []
        for length in self.LENGTHS:
            for volume in self.VOLUMES:
                params = self._grid_params(style, length, volume)
                sample = self._build_sample(
                    name=f"extra_{style}_{length}_{volume}_{len(extras)}",
                    style=style,
                    length=length,
                    volume=volume,
                    conversion_type="CURVES",
                    params=params,
                    source="on_demand"
                )
                extras.append(sample)
                if len(matched) + len(extras) >= count:
                    break
            if len(matched) + len(extras) >= count:
                break

        return (matched + extras)[:count]

    def export_samples_to_json(self, filepath: str, **kwargs) -> int:
        """
        Serialize all samples to a JSON file. Returns sample count.
        Useful for offline inspection or caching.
        """
        samples = self.get_all_samples(**kwargs)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "samples": samples
            }, f, indent=2)
        return len(samples)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_sample(
        self,
        name: str,
        style: str,
        length: str,
        volume: str,
        conversion_type: str,
        params: Dict,
        source: str
    ) -> Dict:
        return {
            "name": name,
            "style": style,
            "length": length,
            "volume": volume,
            "has_hair": True,
            "conversion_type": conversion_type,
            "parameters": params,
            "source": source,
        }

    def _jitter_params(self, params: Dict) -> Dict:
        """Apply random ±JITTER_RANGE noise to every numeric parameter."""
        jittered = {}
        for k, v in params.items():
            if k in self.JITTER_PARAMS and isinstance(v, (int, float)):
                noise = random.uniform(-self.JITTER_RANGE, self.JITTER_RANGE)
                jittered[k] = round(max(0.0, min(1.0, v + noise)), 4)
            else:
                jittered[k] = v
        return jittered

    def _grid_params(self, style: str, length: str, volume: str) -> Dict:
        """Deterministic parameter set derived from style/length/volume axes."""
        length_v  = self.LENGTH_VALUES.get(length, 0.5)
        density_v = self.VOLUME_DENSITY.get(volume, 0.7)

        # Style-specific curl / noise / thickness
        style_defaults = {
            "straight":   {"curl": 0.0,  "noise": 0.15, "thickness": 0.40},
            "wavy":       {"curl": 0.40, "noise": 0.30, "thickness": 0.48},
            "curly":      {"curl": 0.80, "noise": 0.40, "thickness": 0.55},
            "kinky":      {"curl": 1.0,  "noise": 0.55, "thickness": 0.65},
            "afro":       {"curl": 1.0,  "noise": 0.60, "thickness": 0.70},
            "braided":    {"curl": 0.0,  "noise": 0.10, "thickness": 0.55, "braid_intensity": 0.8, "twist": 1.0},
            "dreadlocks": {"curl": 0.15, "noise": 0.20, "thickness": 0.80, "twist": 1.0},
            "pixie":      {"curl": 0.0,  "noise": 0.12, "thickness": 0.35},
            "bun":        {"curl": 0.05, "noise": 0.10, "thickness": 0.42, "bun_size": 0.70, "style_type": 2.0},
            "ponytail":   {"curl": 0.05, "noise": 0.12, "thickness": 0.42, "ponytail_height": 0.60, "style_type": 1.0},
            "undercut":   {"curl": 0.0,  "noise": 0.10, "thickness": 0.38, "sides": 0.0, "top": 0.70, "style_type": 5.0},
            "spiky":      {"curl": 0.0,  "noise": 0.08, "thickness": 0.40, "spike_height": 0.70, "spike_density": 0.60, "style_type": 7.0},
            "quiff":      {"curl": 0.08, "noise": 0.14, "thickness": 0.42, "front_volume": 0.82, "style_type": 6.0},
            "mohawk":     {"curl": 0.0,  "noise": 0.06, "thickness": 0.38, "mohawk_width": 0.70, "style_type": 3.0},
            "cornrows":   {"curl": 0.0,  "noise": 0.05, "thickness": 0.40, "braid_intensity": 1.0, "pattern": 0.50},
            "twisted":    {"curl": 0.30, "noise": 0.20, "thickness": 0.58, "twist": 0.80},
        }

        base = style_defaults.get(style, {"curl": 0.3, "noise": 0.2, "thickness": 0.45})
        params = dict(base)
        params["length"]  = round(length_v, 4)
        params["density"] = round(density_v, 4)
        params["shine"]   = round(max(0.0, 0.7 - params.get("curl", 0.0) * 0.5), 4)
        return params


# =============================================================================
# HAIR PROPERTY ANALYZER (original class – preserved, extended)
# =============================================================================

class HairPropertyAnalyzer:
    """Analyzes text prompts for hair-related features and maps to GBH parameters"""

    def __init__(self):
        # Hair style keywords mapping
        self.hair_style_keywords = {
            'straight': ['straight', 'sleek', 'flat', 'pin straight', 'stick straight'],
            'wavy': ['wavy', 'beachy waves', 'loose waves', 'soft waves', 'undulating'],
            'curly': ['curly', 'curls', 'ringlets', 'coily', 'corkscrew', 'tight curls'],
            'kinky': ['kinky', 'coily', 'tight coils', 'natural hair'],
            'braided': ['braided', 'braids', 'cornrows', 'dutch braid', 'french braid'],
            'twisted': ['twisted', 'dreads', 'locs', 'twists', 'twist out'],
            'bob': ['bob cut', 'bob hairstyle', 'chin-length', 'shoulder-length bob'],
            'pixie': ['pixie cut', 'short crop', 'short hair', 'cropped'],
            'bun': ['bun', 'top knot', 'man bun', 'hair bun'],
            'ponytail': ['ponytail', 'high ponytail', 'low ponytail', 'side ponytail'],
            'mohawk': ['mohawk', 'faux hawk', 'fohawk'],
            'bald': ['bald', 'shaved head', 'no hair', 'bald head'],
            'long': ['long hair', 'flowing hair', 'waist-length', 'very long'],
            'short': ['short hair', 'cropped', 'close cut', 'buzz cut', 'military cut'],
            'beard': ['beard', 'facial hair', 'goatee', 'mustache', 'stubble'],
            'afro': ['afro', 'fro', 'natural afro', 'big hair'],
            'cornrows': ['cornrows', 'rows', 'canerows'],
            'dreadlocks': ['dreadlocks', 'dreads', 'locs', 'sisterlocks'],
            'mullet': ['mullet', 'business in front party in back'],
            'undercut': ['undercut', 'shaved sides', 'fade'],
            'quiff': ['quiff', 'pompadour'],
            'spiky': ['spiky', 'spikes', 'spiked hair'],
        }

        self.length_keywords = {
            'very_short': ['buzz cut', 'shaved', 'very short', 'military cut'],
            'short': ['short hair', 'cropped', 'ear-length'],
            'medium': ['medium length', 'shoulder-length', 'collarbone-length'],
            'long': ['long hair', 'chest-length', 'waist-length'],
            'very_long': ['very long', 'hip-length', 'floor-length', 'extremely long'],
        }

        self.volume_keywords = {
            'thin': ['thin hair', 'fine hair', 'sparse', 'thin', 'low density'],
            'normal': ['normal hair', 'average thickness', 'medium density'],
            'thick': ['thick hair', 'full hair', 'voluminous', 'dense', 'lots of hair'],
            'very_thick': ['very thick', 'extremely thick', 'super dense', 'very full'],
        }

        self.color_keywords = {
            'black': ['black hair', 'jet black', 'dark black', 'raven hair'],
            'brown': ['brown hair', 'brunette', 'chestnut', 'chocolate brown'],
            'blonde': ['blonde', 'blond', 'golden hair', 'platinum', 'ash blonde'],
            'red': ['red hair', 'ginger', 'auburn', 'copper', 'strawberry blonde'],
            'gray': ['gray hair', 'grey hair', 'silver', 'white hair', 'salt and pepper'],
            'blue': ['blue hair', 'dyed blue', 'cyan hair'],
            'pink': ['pink hair', 'rose gold', 'magenta hair'],
            'purple': ['purple hair', 'lavender hair', 'violet hair'],
            'green': ['green hair', 'emerald hair', 'teal hair'],
            'rainbow': ['rainbow hair', 'multicolor', 'colorful hair'],
        }

        self.gbh_conversion_types = {
            'strands': 'CURVES',
            'mesh': 'MESH',
            'cards': 'MESH',
            'particles': 'PARTICLES',
        }

        self.intensity_modifiers = {
            'slightly': 0.3, 'somewhat': 0.4, 'moderately': 0.6,
            'very': 0.8, 'extremely': 0.9, 'incredibly': 1.0,
            'quite': 0.7, 'rather': 0.6, 'fairly': 0.5
        }

        self.default_intensity = 0.65

        # Shared sample generator instance
        self._sample_generator = HairSampleGenerator()

    # ── Prompt analysis (unchanged from original) ─────────────────────────────

    def analyze_hair_prompt(self, prompt: str) -> Dict[str, Any]:
        """Main method to analyze hair-related text and return GBH parameters"""
        prompt_lower = prompt.lower()

        hair_analysis = {
            'has_hair': True,
            'style': None,
            'length': None,
            'volume': None,
            'color': None,
            'conversion_type': 'CURVES',
            'special_features': [],
            'parameters': {},
            'confidence': 0.0
        }

        for kw in self.hair_style_keywords['bald']:
            if kw in prompt_lower:
                hair_analysis['has_hair'] = False
                hair_analysis['style'] = 'bald'
                return hair_analysis

        detected_style  = self._detect_feature(prompt_lower, self.hair_style_keywords)
        detected_length = self._detect_feature(prompt_lower, self.length_keywords)
        detected_volume = self._detect_feature(prompt_lower, self.volume_keywords)
        detected_color  = self._detect_feature(prompt_lower, self.color_keywords)

        if detected_style:  hair_analysis['style']  = detected_style
        if detected_length: hair_analysis['length'] = detected_length
        if detected_volume: hair_analysis['volume'] = detected_volume
        if detected_color:  hair_analysis['color']  = detected_color

        if any(x in prompt_lower for x in ['mesh hair', 'stylized', 'cartoon', 'anime']):
            hair_analysis['conversion_type'] = 'MESH'
        elif any(x in prompt_lower for x in ['hair cards', 'game', 'low poly']):
            hair_analysis['conversion_type'] = 'MESH'
        elif any(x in prompt_lower for x in ['particles', 'particle system']):
            hair_analysis['conversion_type'] = 'PARTICLES'

        hair_analysis['parameters'] = self._map_to_gbh_parameters(hair_analysis, prompt_lower)

        # Resolve color RGBA
        hair_analysis['color_rgba'] = self._resolve_color_rgba(
            detected_color, prompt_lower
        )

        detected_count = sum(
            1 for v in [detected_style, detected_length, detected_volume, detected_color] if v
        )
        hair_analysis['confidence'] = detected_count / 4.0 if detected_count > 0 else 0.0

        return hair_analysis

    def _detect_feature(self, prompt_lower: str, keyword_dict: Dict) -> Optional[str]:
        best_match = None
        best_score = 0

        for feature, keywords in keyword_dict.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    for modifier, mod_score in self.intensity_modifiers.items():
                        if f"{modifier} {keyword}" in prompt_lower:
                            score += mod_score * 2
                            break
                    else:
                        score += self.default_intensity

                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, prompt_lower):
                        score += 0.3

            if score > best_score:
                best_score = score
                best_match = feature

        return best_match if best_score > 0 else None

    def _resolve_color_rgba(
        self,
        color_label: Optional[str],
        prompt_lower: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return an RGBA tuple for the detected colour, or None."""
        # Check explicit color keyword map first (longest match wins)
        for kw in sorted(COLOR_KEYWORD_MAP.keys(), key=len, reverse=True):
            if kw in prompt_lower:
                preset_name = COLOR_KEYWORD_MAP[kw]
                return HAIR_COLOR_PRESETS.get(preset_name)

        # Fall back to the coarser color_label from analysis
        if color_label:
            label_map = {
                "black": "black", "brown": "brown", "blonde": "blonde",
                "red": "red", "gray": "gray", "blue": "blue",
                "pink": "pink", "purple": "purple", "green": "green",
            }
            preset_name = label_map.get(color_label)
            if preset_name:
                return HAIR_COLOR_PRESETS.get(preset_name)

        return None

    def _map_to_gbh_parameters(self, analysis: Dict, prompt_lower: str) -> Dict[str, float]:
        """Map detected features to GBH parameters"""
        params = {}

        style_params = {
            'straight': {'curl': 0.0,  'noise': 0.2,  'density': 0.7},
            'wavy':     {'curl': 0.4,  'noise': 0.3,  'density': 0.7},
            'curly':    {'curl': 0.8,  'noise': 0.4,  'density': 0.8},
            'kinky':    {'curl': 1.0,  'noise': 0.5,  'density': 0.9},
            'afro':     {'curl': 1.0,  'noise': 0.6,  'density': 1.0,  'volume': 1.0},
            'braided':  {'braid_intensity': 0.8, 'twist': 1.0, 'density': 0.8},
            'twisted':  {'twist': 0.8, 'curl': 0.3, 'density': 0.7},
            'dreadlocks': {'twist': 1.0, 'curl': 0.2, 'density': 0.6, 'thickness': 0.8},
            'cornrows': {'braid_intensity': 1.0, 'pattern': 0.5, 'density': 0.7},
            'bob':      {'length': 0.3, 'volume': 0.5, 'style_type': 0.5},
            'pixie':    {'length': 0.1, 'volume': 0.3, 'style_type': 0.3},
            'bun':      {'bun_size': 0.7, 'style_type': 2.0, 'length': 0.2},
            'ponytail': {'ponytail_height': 0.6, 'style_type': 1.0, 'length': 0.6},
            'mohawk':   {'mohawk_width': 0.7, 'style_type': 3.0, 'length': 0.4},
            'mullet':   {'length_front': 0.2, 'length_back': 0.8, 'style_type': 4.0},
            'undercut': {'sides': 0.0, 'top': 0.7, 'style_type': 5.0},
            'quiff':    {'front_volume': 0.8, 'length': 0.5, 'style_type': 6.0},
            'spiky':    {'spike_height': 0.7, 'spike_density': 0.6, 'style_type': 7.0},
        }

        length_to_value = {
            'very_short': 0.1, 'short': 0.3, 'medium': 0.5,
            'long': 0.7, 'very_long': 0.9,
        }

        volume_to_value = {
            'thin': 0.3, 'normal': 0.5, 'thick': 0.7, 'very_thick': 0.9,
        }

        style = analysis.get('style')
        if style and style in style_params:
            params.update(style_params[style])

        if analysis['length'] and analysis['length'] in length_to_value:
            params['length'] = length_to_value[analysis['length']]

        if analysis['volume'] and analysis['volume'] in volume_to_value:
            params['density'] = volume_to_value[analysis['volume']]
            params['volume']  = volume_to_value[analysis['volume']]

        if 'middle part' in prompt_lower or 'center part' in prompt_lower:
            params['parting'] = 0.5
        elif 'side part' in prompt_lower or 'left part' in prompt_lower:
            params['parting'] = 0.3
        elif 'right part' in prompt_lower:
            params['parting'] = 0.7

        if any(x in prompt_lower for x in ['bangs', 'fringe', 'front hair']):
            params['bangs'] = 0.7
            params['bangs_length'] = params.get('length', 0.5) * 0.7

        if any(x in prompt_lower for x in ['beard', 'mustache', 'goatee', 'stubble']):
            params['facial_hair'] = 1.0
            if 'stubble' in prompt_lower:
                params['beard_length'] = 0.1
            elif 'goatee' in prompt_lower:
                params['beard_style'] = 0.5
            elif 'mustache' in prompt_lower:
                params['mustache'] = 1.0

        return params

    def get_gbh_operator_sequence(
        self, target_object_name: str, hair_analysis: Dict
    ) -> List[Dict]:
        """Generate sequence of GBH operators to execute in Blender"""
        operators = []

        if not hair_analysis.get('has_hair', True):
            return []

        operators.append({'type': 'SELECT', 'object': target_object_name})

        conversion_type = hair_analysis.get('conversion_type', 'CURVES')

        if conversion_type == 'PARTICLES':
            operators.append({
                'type': 'GBH_OT_strands_to_particle',
                'properties': {}
            })
        else:
            operators.append({
                'type': 'GBH_OT_convert_hair',
                'properties': {'convert_to': 'CURVES'}
            })
            if conversion_type == 'MESH':
                operators.append({
                    'type': 'GBH_OT_convert_hair',
                    'properties': {'convert_to': 'MESH'}
                })

        if hair_analysis.get('attach_to_surface', False):
            operators.append({
                'type': 'GBH_OT_attach_curves_to_surface',
                'properties': {}
            })

        return operators

    # ── NEW: Sample-based generation helpers ──────────────────────────────────

    def get_sample_by_name(self, name: str) -> Optional[Dict]:
        """
        Retrieve a single named sample from the archetype library.
        Falls back to fuzzy style match if name not found.
        """
        if name in HAIR_ARCHETYPES:
            arch = HAIR_ARCHETYPES[name]
            return {
                "name": name,
                "style": arch["style"],
                "length": arch["length"],
                "volume": arch.get("volume", "normal"),
                "has_hair": True,
                "conversion_type": arch.get("conversion_type", "CURVES"),
                "parameters": dict(arch["params"]),
                "source": "archetype",
            }

        # Try partial match
        name_lower = name.lower()
        for arch_name, arch in HAIR_ARCHETYPES.items():
            if name_lower in arch_name or arch_name in name_lower:
                return self.get_sample_by_name(arch_name)

        return None

    def get_samples_for_prompt(
        self,
        prompt: str,
        count: int = 5,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Given a text prompt, return `count` sample dicts whose style/length/volume
        best match the prompt, with random variation.

        This is the main entry point for programmatic batch generation.
        """
        if seed is not None:
            random.seed(seed)

        analysis = self.analyze_hair_prompt(prompt)

        if not analysis.get('has_hair', True):
            return [{"has_hair": False, "style": "bald", "parameters": {}, "name": "bald"}]

        style  = analysis.get("style")
        length = analysis.get("length")
        volume = analysis.get("volume")
        color  = analysis.get("color")
        color_rgba = analysis.get("color_rgba")

        # Collect candidate samples
        gen = self._sample_generator
        all_samples = gen.get_all_samples()

        def score(s: Dict) -> int:
            sc = 0
            if style  and s.get("style")  == style:  sc += 3
            if length and s.get("length") == length: sc += 2
            if volume and s.get("volume") == volume: sc += 1
            return sc

        ranked = sorted(all_samples, key=score, reverse=True)
        top = ranked[:max(count * 3, 12)]

        # Pick `count` with slight randomness (weighted shuffle)
        selected = random.sample(top, min(count, len(top)))

        # Inject color override if detected
        if color_rgba:
            for s in selected:
                s = dict(s)           # shallow copy
                p = dict(s["parameters"])
                p["color_r"] = color_rgba[0]
                p["color_g"] = color_rgba[1]
                p["color_b"] = color_rgba[2]
                p["color_a"] = color_rgba[3]
                s["parameters"] = p
                s["color"]      = color
                s["color_rgba"] = color_rgba

        return selected

    def get_all_programmatic_samples(
        self,
        variations_per_archetype: int = 4,
        include_grid: bool = True,
        include_colors: bool = True
    ) -> List[Dict]:
        """
        Expose the full sample library for external use (e.g. Blender bridge).
        """
        return self._sample_generator.get_all_samples(
            variations_per_archetype=variations_per_archetype,
            include_grid=include_grid,
            include_colors=include_colors,
        )

    def export_samples(self, filepath: str) -> int:
        """Export all samples to JSON. Returns count."""
        return self._sample_generator.export_samples_to_json(filepath)


# =============================================================================
# GLOBAL INSTANCE (backward-compatible)
# =============================================================================

hair_analyzer = HairPropertyAnalyzer()

# Convenience: shared generator for direct import
sample_generator = HairSampleGenerator()