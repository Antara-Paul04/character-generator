import bpy
import re
import json
import os
import traceback
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional

# =============================================================================
# BLENDER BRIDGE CONFIGURATION
# =============================================================================
COMMUNICATION_DIR = r"C:\temp\blender_bridge"
REQUEST_FILE      = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE     = os.path.join(COMMUNICATION_DIR, "character_response.json")
HEARTBEAT_FILE    = os.path.join(COMMUNICATION_DIR, "blender_heartbeat.json")

# ---------------------------------------------------------------------------
# Hair analyzer import
# ---------------------------------------------------------------------------
try:
    import sys
    possible_paths = [
        os.path.dirname(os.path.dirname(__file__)),
        r"E:\work\work\sem_project\character-generator",
        os.getcwd()
    ]
    for path in possible_paths:
        if path not in sys.path:
            sys.path.append(path)

    from hair_generator import hair_analyzer, sample_generator, HAIR_ARCHETYPES, HAIR_COLOR_PRESETS
    HAIR_SUPPORT = True
    print("✓ Hair generator module loaded successfully")
    print(f"  Archetype library: {len(HAIR_ARCHETYPES)} styles")
    print(f"  Color presets    : {len(HAIR_COLOR_PRESETS)}")

except ImportError as e:
    print(f"⚠️ Hair generator module not available: {e}")
    HAIR_SUPPORT = False

    class _DummyHairAnalyzer:
        def analyze_hair_prompt(self, prompt):
            return {'has_hair': True, 'style': None, 'length': None,
                    'color': None, 'parameters': {}}
        def get_gbh_operator_sequence(self, obj, analysis):
            return []
        def get_samples_for_prompt(self, prompt, count=5, seed=None):
            return []
        def get_all_programmatic_samples(self, **kw):
            return []
        def export_samples(self, filepath):
            return 0

    class _DummySampleGenerator:
        def get_all_samples(self, **kw):     return []
        def get_archetype_samples(self):     return []
        def get_variation_samples(self, n=4):return []
        def get_grid_samples(self):          return []
        def get_color_samples(self):         return []
        def export_samples_to_json(self, fp, **kw): return 0

    HAIR_ARCHETYPES    = {}
    HAIR_COLOR_PRESETS = {}
    hair_analyzer      = _DummyHairAnalyzer()
    sample_generator   = _DummySampleGenerator()

# ---------------------------------------------------------------------------
# GBH Tool detection
# ---------------------------------------------------------------------------
GBH_AVAILABLE = False
try:
    if hasattr(bpy.context.preferences, 'addons'):
        addon_names = [a.module for a in bpy.context.preferences.addons]
        gbh_addons  = [n for n in addon_names if 'gbh' in n.lower()]
        if gbh_addons:
            print(f"✓ GBH Tool found: {gbh_addons}")
            GBH_AVAILABLE = True
        else:
            print("⚠️ GBH Tool not found in enabled addons")
except Exception as e:
    print(f"⚠️ GBH Tool check failed: {e}")

# ---------------------------------------------------------------------------
# Bridge state
# ---------------------------------------------------------------------------
is_monitoring   = False
last_heartbeat  = None
failed_requests = 0
max_failures    = 3


# =============================================================================
# HEARTBEAT
# =============================================================================
def update_heartbeat():
    global last_heartbeat
    try:
        heartbeat = {
            "timestamp":     datetime.now().isoformat(),
            "pid":           os.getpid(),
            "gbh_available": GBH_AVAILABLE,
            "hair_support":  HAIR_SUPPORT,
            "status":        "running"
        }
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump(heartbeat, f)
        last_heartbeat = datetime.now()
    except Exception:
        pass
    return 5.0


# =============================================================================
# GBH OPERATOR DISCOVERY
# =============================================================================
def find_gbh_operators():
    """Dynamically find all GBH-related operators in Blender"""
    gbh_ops = {'generate': [], 'remove': [], 'style': [], 'property': []}
    try:
        for category in dir(bpy.ops):
            if category.startswith('_'):
                continue
            try:
                ops_module = getattr(bpy.ops, category)
                for op_name in dir(ops_module):
                    if op_name.startswith('_'):
                        continue
                    full_op = f"{category}.{op_name}"
                    if 'gbh' in full_op.lower() or 'hair' in full_op.lower():
                        if any(k in op_name.lower() for k in ('generate', 'create', 'add')):
                            gbh_ops['generate'].append(full_op)
                        elif any(k in op_name.lower() for k in ('remove', 'delete')):
                            gbh_ops['remove'].append(full_op)
                        elif any(k in op_name.lower() for k in ('style', 'set')):
                            gbh_ops['style'].append(full_op)
                        else:
                            gbh_ops['property'].append(full_op)
            except Exception:
                continue
    except Exception as e:
        print(f"Error scanning operators: {e}")
    return gbh_ops


# =============================================================================
# HAIR GENERATION – CORE
# =============================================================================

def _apply_gbh_params_to_object(character_obj, params: Dict):
    """
    Best-effort attempt to set GBH hair parameters on a curves / hair object.
    Works whether called from a live generation or a sample playback.
    """
    if not GBH_AVAILABLE:
        return

    gbh_ops = find_gbh_operators()

    # Try property-setter operators
    for op_name in gbh_ops.get('property', []):
        try:
            category, op = op_name.split('.')
            op_func = getattr(getattr(bpy.ops, category), op)
            filtered = {k: v for k, v in params.items()
                        if isinstance(v, (int, float)) and v is not None}
            try:
                op_func(**filtered)
                return
            except TypeError:
                op_func()
                return
        except Exception:
            continue


def _try_generate_hair(character_obj, params: Dict, conversion_type: str) -> bool:
    """
    Attempt hair creation via GBH generation operators.
    Returns True on success.
    """
    gbh_ops = find_gbh_operators()
    bpy.context.view_layer.objects.active = character_obj
    character_obj.select_set(True)

    for op_name in gbh_ops.get('generate', []):
        try:
            category, op = op_name.split('.')
            op_func = getattr(getattr(bpy.ops, category), op)

            call_params = {
                'target':     character_obj.name,
                'hair_type':  'strands',
                'length':     params.get('length', 0.5),
                'curl':       params.get('curl', 0.3),
                'density':    params.get('density', 0.7),
            }
            call_params = {k: v for k, v in call_params.items() if v is not None}

            try:
                op_func(**call_params)
            except TypeError:
                op_func()

            print(f"  ✓ Hair generated via {op_name}")
            return True

        except Exception as e:
            print(f"  ✗ {op_name} failed: {e}")
            continue

    return False


def generate_hair_with_gbh(character_obj, hair_analysis: Dict) -> Dict:
    """
    Generate hair using GBH Tool.
    Accepts either a prompt-analysis dict or a pre-built sample dict.
    """
    global failed_requests

    if not GBH_AVAILABLE:
        print("⚠️ GBH Tool not available – skipping hair generation")
        return {"success": False, "reason": "GBH Tool not installed/enabled"}

    if not hair_analysis.get('has_hair', True):
        print("✓ No hair requested (bald/balding detected)")
        return {"success": True, "hair_generated": False, "reason": "bald"}

    params          = hair_analysis.get('parameters', {})
    conversion_type = hair_analysis.get('conversion_type', 'CURVES')
    style           = hair_analysis.get('style', 'unknown')
    color_rgba      = hair_analysis.get('color_rgba')

    print(f"\n{'='*60}")
    print(f"💇 GENERATING HAIR  |  style={style}  "
          f"length={hair_analysis.get('length','?')}  "
          f"volume={hair_analysis.get('volume','?')}")
    print(f"   parameters: {params}")
    if color_rgba:
        print(f"   color RGBA : {color_rgba}")
    print(f"{'='*60}")

    success = _try_generate_hair(character_obj, params, conversion_type)

    if success:
        failed_requests = 0
        # Also push property values after generation
        _apply_gbh_params_to_object(character_obj, params)
        return {
            "success":        True,
            "hair_generated": True,
            "style":          style,
            "parameters":     params,
            "color_rgba":     color_rgba,
        }

    # If generate operators are absent, try property operators on any existing hair
    print("⚠️ No generator operators worked – attempting property-only path")
    gbh_ops = find_gbh_operators()
    if gbh_ops.get('property'):
        _apply_gbh_params_to_object(character_obj, params)
        return {"success": True, "hair_generated": True, "style": style, "reason": "property_only"}

    failed_requests += 1
    return {"success": False, "reason": "all_operators_failed"}


def apply_hair_from_analysis(character_obj, hair_analysis: Dict) -> Dict:
    """Wrapper for hair generation with error handling"""
    try:
        return generate_hair_with_gbh(character_obj, hair_analysis)
    except Exception as e:
        print(f"✗ Hair generation error: {e}")
        traceback.print_exc()
        return {"success": False, "reason": str(e), "hair_generated": False}


# =============================================================================
# PROGRAMMATIC HAIR SAMPLE GENERATION (NEW)
# =============================================================================

def generate_hair_sample_batch(
    character_obj,
    prompt: str         = "",
    count: int          = 5,
    strategy: str       = "prompt",   # "prompt" | "archetype" | "grid" | "all"
    seed: Optional[int] = None,
    export_path: str    = ""
) -> List[Dict]:
    """
    Generate multiple hair samples programmatically on the character object.

    Parameters
    ----------
    character_obj : bpy Object
        The Blender mesh to apply hair to.
    prompt : str
        Original text prompt (used by 'prompt' strategy).
    count : int
        How many samples to generate (used by 'prompt' / 'archetype').
    strategy : str
        'prompt'    – pick samples that match the text prompt (default)
        'archetype' – cycle through the named archetype library
        'grid'      – systematic style × length sweep
        'all'       – every sample in the library
    seed : int | None
        Random seed for reproducibility.
    export_path : str
        Optional JSON file path to save the batch manifest.

    Returns
    -------
    List of result dicts, one per sample applied.
    """
    if not HAIR_SUPPORT:
        print("⚠️ Hair support not available – skipping batch generation")
        return []

    if seed is not None:
        import random
        random.seed(seed)

    # ── Build the sample list ────────────────────────────────────────────────
    if strategy == "prompt":
        samples = hair_analyzer.get_samples_for_prompt(prompt, count=count, seed=seed)

    elif strategy == "archetype":
        raw = sample_generator.get_archetype_samples()
        import random as _rnd
        _rnd.shuffle(raw)
        samples = raw[:count]

    elif strategy == "grid":
        samples = sample_generator.get_grid_samples()[:count]

    elif strategy == "all":
        samples = sample_generator.get_all_samples()

    else:
        print(f"⚠️ Unknown strategy '{strategy}', falling back to 'prompt'")
        samples = hair_analyzer.get_samples_for_prompt(prompt, count=count, seed=seed)

    if not samples:
        print("⚠️ No samples to generate")
        return []

    print(f"\n{'='*70}")
    print(f"🎨 PROGRAMMATIC HAIR BATCH  |  strategy={strategy}  "
          f"samples={len(samples)}")
    print(f"{'='*70}")

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n  [{i}/{len(samples)}] {sample.get('name', 'unnamed')} "
              f"  style={sample.get('style','?')}  "
              f"length={sample.get('length','?')}  "
              f"volume={sample.get('volume','?')}")

        result = apply_hair_from_analysis(character_obj, sample)
        result["sample_name"]   = sample.get("name", "unnamed")
        result["sample_source"] = sample.get("source", "unknown")
        result["index"]         = i
        results.append(result)

        # Small delay so Blender can breathe between heavy operations
        time.sleep(0.05)

    # ── Optional JSON export ─────────────────────────────────────────────────
    if export_path:
        try:
            os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else ".", exist_ok=True)
            manifest = {
                "generated_at": datetime.now().isoformat(),
                "strategy":     strategy,
                "prompt":       prompt,
                "total":        len(samples),
                "samples":      samples,
                "results":      results,
            }
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"\n✓ Batch manifest saved to {export_path}")
        except Exception as e:
            print(f"⚠️ Could not save manifest: {e}")

    success_count = sum(1 for r in results if r.get("success"))
    print(f"\n✅ Batch complete: {success_count}/{len(samples)} successful")
    return results


def get_hair_sample_stats() -> Dict:
    """Return statistics about the available hair sample library."""
    if not HAIR_SUPPORT:
        return {"available": False}

    all_samples = sample_generator.get_all_samples()
    by_source   = {}
    by_style    = {}
    by_length   = {}

    for s in all_samples:
        src = s.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

        sty = s.get("style", "unknown")
        by_style[sty] = by_style.get(sty, 0) + 1

        lng = s.get("length", "unknown")
        by_length[lng] = by_length.get(lng, 0) + 1

    return {
        "available":       True,
        "total_samples":   len(all_samples),
        "archetypes":      len(HAIR_ARCHETYPES),
        "color_presets":   len(HAIR_COLOR_PRESETS),
        "by_source":       by_source,
        "by_style":        by_style,
        "by_length":       by_length,
    }


# =============================================================================
# BLENDER HELPER FUNCTIONS
# =============================================================================

def get_object_by_gender(gender="male"):
    """Get the appropriate character object with fallback"""
    exact_names = [f"mb_{gender}", f"MB_{gender}", f"Mb_{gender}"]
    for name in exact_names:
        if name in bpy.data.objects:
            obj = bpy.data.objects[name]
            print(f"✓ Found exact match: {obj.name}")
            return obj

    gender_lower = gender.lower()
    for obj in bpy.data.objects:
        if f"mb_{gender_lower}" in obj.name.lower():
            print(f"✓ Found character object: {obj.name}")
            return obj

    fallback = "female" if gender == "male" else "male"
    for obj in bpy.data.objects:
        if f"mb_{fallback}" in obj.name.lower():
            print(f"⚠️ Using fallback {fallback} object: {obj.name}")
            return obj

    for obj in bpy.data.objects:
        if 'mb_' in obj.name.lower():
            print(f"⚠️ Using generic MB object: {obj.name}")
            return obj

    print(f"✗ Could not find any character object for gender: {gender}")
    return None


def reset_character_shape_keys(obj):
    """Reset all shape keys"""
    if not obj:
        return
    if not hasattr(obj.data, "shape_keys") or not obj.data.shape_keys:
        return
    try:
        reset_count = 0
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name != "Basis":
                kb.value = 0.0
                reset_count += 1
        print(f"✓ Reset {reset_count} shape keys")
    except Exception as e:
        print(f"✗ Error resetting shape keys: {e}")


def apply_morph(obj, shape_key_name, value):
    """Safely apply a morph value"""
    if not obj:
        return False
    if not hasattr(obj.data, "shape_keys") or not obj.data.shape_keys:
        return False
    try:
        if shape_key_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[shape_key_name].value = \
                min(1.0, max(0.0, float(value)))
            return True
    except Exception as e:
        print(f"⚠️ Error applying {shape_key_name}: {e}")
    return False


# =============================================================================
# ENHANCED PROPERTY PROCESSING
# =============================================================================

def process_enhanced_properties(structured_data: Dict, character_obj, gender: str):
    """Apply morphs and generate hair with comprehensive error handling"""
    results = {
        "morphs_applied": 0,
        "morphs_failed":  0,
        "morph_details":  [],
        "hair_result":    {"success": False, "hair_generated": False},
        "errors":         []
    }

    try:
        # ── Morphs ────────────────────────────────────────────────────────────
        print(f"\n--- Applying Morphs for {gender.upper()} ---")
        reset_character_shape_keys(character_obj)

        properties = structured_data.get("properties", {})
        print(f"Processing {len(properties)} properties...")

        for prop_name, intensity in properties.items():
            if apply_morph(character_obj, prop_name, intensity):
                results["morphs_applied"] += 1
                results["morph_details"].append(f"{prop_name}: {intensity:.2f}")
            else:
                results["morphs_failed"] += 1

        bpy.context.view_layer.update()
        print(f"✓ Applied {results['morphs_applied']} morphs, "
              f"{results['morphs_failed']} failed")

        # ── Hair ──────────────────────────────────────────────────────────────
        original_prompt = structured_data.get("prompt", "")
        hair_analysis   = structured_data.get("hair_analysis", {})

        # Re-analyse from prompt if analysis is missing
        if not hair_analysis and original_prompt and HAIR_SUPPORT:
            print("\n--- Analysing Hair from Prompt ---")
            hair_analysis = hair_analyzer.analyze_hair_prompt(original_prompt)
            print(f"  Analysis: {hair_analysis}")

        if hair_analysis:
            print("\n--- Generating Hair ---")
            hair_result = apply_hair_from_analysis(character_obj, hair_analysis)
            results["hair_result"] = hair_result
            if hair_result.get("success"):
                print(f"✓ Hair generation successful: "
                      f"{hair_result.get('style', 'unknown')}")
            else:
                print(f"⚠️ Hair generation issue: "
                      f"{hair_result.get('reason', 'unknown')}")
        else:
            print("\n--- No hair analysis available ---")

    except Exception as e:
        msg = f"Error in process_enhanced_properties: {e}"
        print(f"✗ {msg}")
        traceback.print_exc()
        results["errors"].append(msg)

    return results


# =============================================================================
# BRIDGE MONITORING
# =============================================================================

def start_bridge_monitoring():
    global is_monitoring, failed_requests

    if is_monitoring:
        print("Bridge monitoring is already active.")
        return

    os.makedirs(COMMUNICATION_DIR, exist_ok=True)
    failed_requests = 0

    for f in [REQUEST_FILE, RESPONSE_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    stats = get_hair_sample_stats()

    print(f"\n{'='*70}")
    print(f"🎭 ENHANCED BLENDER BRIDGE  |  HAIR BATCH GENERATION READY")
    print(f"{'='*70}")
    print(f"📁 Watch directory : {COMMUNICATION_DIR}")
    print(f"💇 GBH Tool        : {'✓ AVAILABLE' if GBH_AVAILABLE else '✗ NOT FOUND'}")
    print(f"🧬 Hair support    : {'✓ LOADED' if HAIR_SUPPORT else '✗ NOT AVAILABLE'}")
    if stats.get("available"):
        print(f"📚 Sample library  : {stats['total_samples']} samples  "
              f"({stats['archetypes']} archetypes, "
              f"{stats['color_presets']} colors)")
    print(f"⚡ Max failures    : {max_failures}")
    print(f"{'='*70}\n")

    is_monitoring = True

    if not bpy.app.timers.is_registered(check_for_requests):
        bpy.app.timers.register(check_for_requests, first_interval=1.0)

    if not bpy.app.timers.is_registered(update_heartbeat):
        bpy.app.timers.register(update_heartbeat, first_interval=1.0)

    print("✅ Bridge monitoring started")


def stop_bridge_monitoring():
    global is_monitoring

    is_monitoring = False

    for fn in [check_for_requests, update_heartbeat]:
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)

    try:
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump({"timestamp": datetime.now().isoformat(), "status": "stopped"}, f)
    except Exception:
        pass

    print("Bridge monitoring stopped.")


def check_for_requests():
    global is_monitoring, failed_requests

    if not is_monitoring:
        return None

    if failed_requests >= max_failures:
        print(f"\n⚠️ Too many failures ({failed_requests}). Auto-stopping bridge.")
        stop_bridge_monitoring()
        return None

    try:
        if not os.path.exists(REQUEST_FILE):
            return 0.5

        with open(REQUEST_FILE, 'r', encoding='utf-8') as f:
            request_data = json.load(f)

        structured_data = request_data.get('structured_data', {})
        prompt          = request_data.get('prompt', '')

        # ── Hair batch mode ───────────────────────────────────────────────────
        if request_data.get('mode') == 'hair_batch':
            _handle_hair_batch_request(request_data, prompt)
            try:
                os.remove(REQUEST_FILE)
            except Exception:
                pass
            return 0.5

        # ── Normal character generation mode ──────────────────────────────────
        if not structured_data:
            print("✗ Error: Missing structured_data")
            send_error_response(prompt, "Missing structured_data")
            os.remove(REQUEST_FILE)
            failed_requests += 1
            return 0.5

        gender    = structured_data.get('gender', 'male')
        character = get_object_by_gender(gender)

        if character:
            results = process_enhanced_properties(structured_data, character, gender)

            response_data = {
                "timestamp":       datetime.now().isoformat(),
                "prompt":          prompt,
                "gender":          gender,
                "status":          "completed",
                "message":         f"✓ {gender.capitalize()} character generated",
                "morphs_applied":  results["morphs_applied"],
                "morphs_failed":   results["morphs_failed"],
                "hair_generated":  results["hair_result"].get("hair_generated", False),
                "hair_style":      results["hair_result"].get("style", "unknown"),
                "hair_success":    results["hair_result"].get("success", False),
                "character_object":character.name,
                "errors":          results["errors"]
            }

            with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2)

            print(f"\n✅ Response sent – morphs: {results['morphs_applied']}, "
                  f"hair: {results['hair_result'].get('hair_generated', False)}")
            failed_requests = 0

        else:
            error_msg = f"Could not find {gender} character object"
            print(f"✗ {error_msg}")
            send_error_response(prompt, error_msg, gender)
            failed_requests += 1

        try:
            os.remove(REQUEST_FILE)
        except Exception:
            pass

    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in request file: {e}")
        failed_requests += 1
        try:
            os.remove(REQUEST_FILE)
        except Exception:
            pass

    except Exception as e:
        print(f"✗ Error processing request: {e}")
        traceback.print_exc()
        failed_requests += 1
        try:
            send_error_response("Unknown", f"Error: {str(e)}")
            if os.path.exists(REQUEST_FILE):
                os.remove(REQUEST_FILE)
        except Exception:
            pass

    return 0.5


def _handle_hair_batch_request(request_data: Dict, prompt: str):
    """Process a dedicated hair-batch request from frontend."""
    gender   = request_data.get('gender', 'male')
    strategy = request_data.get('strategy', 'prompt')
    count    = request_data.get('count', 5)
    seed     = request_data.get('seed', None)
    export   = request_data.get('export_path', '')

    character = get_object_by_gender(gender)
    if not character:
        send_error_response(prompt, f"No {gender} character found", gender)
        return

    print(f"\n{'='*60}")
    print(f"🎨 HAIR BATCH REQUEST  |  strategy={strategy}  count={count}")
    print(f"{'='*60}")

    batch_results = generate_hair_sample_batch(
        character_obj=character,
        prompt=prompt,
        count=count,
        strategy=strategy,
        seed=seed,
        export_path=export
    )

    success_count = sum(1 for r in batch_results if r.get("success"))

    response_data = {
        "timestamp":       datetime.now().isoformat(),
        "prompt":          prompt,
        "gender":          gender,
        "status":          "completed",
        "mode":            "hair_batch",
        "message":         f"✓ {success_count}/{len(batch_results)} hair samples generated",
        "strategy":        strategy,
        "total_samples":   len(batch_results),
        "success_count":   success_count,
        "batch_results":   batch_results,
        "character_object":character.name,
    }

    with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=2)

    print(f"✅ Batch response sent – {success_count}/{len(batch_results)} ok")


def send_error_response(prompt, error_message, gender="unknown"):
    """Helper to send error responses"""
    try:
        error_response = {
            "timestamp":      datetime.now().isoformat(),
            "prompt":         prompt,
            "gender":         gender,
            "status":         "error",
            "message":        error_message,
            "morphs_applied": 0,
            "hair_generated": False
        }
        with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
            json.dump(error_response, f, indent=2)
    except Exception:
        pass


# =============================================================================
# BLENDER UI PANEL
# =============================================================================

class MESH_PT_character_bridge(bpy.types.Panel):
    bl_label      = "Character Generator Bridge"
    bl_idname     = "MESH_PT_character_bridge"
    bl_space_type = 'PROPERTIES'
    bl_region_type= 'WINDOW'
    bl_context    = "object"

    def draw(self, context):
        layout = self.layout

        # Status
        box = layout.box()
        box.label(text="BRIDGE STATUS", icon='INFO')
        box.label(
            text="● RUNNING" if is_monitoring else "○ STOPPED",
            icon='CHECKMARK' if is_monitoring else 'X'
        )
        box.label(
            text="✓ GBH Tool: Available" if GBH_AVAILABLE else "✗ GBH Tool: Not Found",
            icon='CHECKMARK' if GBH_AVAILABLE else 'ERROR'
        )
        box.label(
            text="✓ Hair Analyzer: Loaded" if HAIR_SUPPORT else "✗ Hair Analyzer: Not Loaded",
            icon='CHECKMARK' if HAIR_SUPPORT else 'ERROR'
        )

        # Sample library stats
        if HAIR_SUPPORT:
            stats = get_hair_sample_stats()
            if stats.get("available"):
                box.label(text=f"  Library: {stats['total_samples']} samples  "
                               f"({stats['archetypes']} archetypes)")

        # Control
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        if is_monitoring:
            row.operator("mesh.stop_bridge",  text="Stop Bridge",  icon='PAUSE')
        else:
            row.operator("mesh.start_bridge", text="Start Bridge", icon='PLAY')

        # Character tests
        layout.separator()
        box = layout.box()
        box.label(text="TEST GENERATION", icon='EXPERIMENTAL')
        row = box.row(align=True)
        row.operator("mesh.test_male",   text="Test Male")
        row.operator("mesh.test_female", text="Test Female")

        # Hair tests
        layout.separator()
        box = layout.box()
        box.label(text="HAIR SAMPLES", icon='HAIR')

        row = box.row(align=True)
        row.operator("mesh.test_hair",          text="Test Hair (selected)")
        row.operator("mesh.test_hair_batch",    text="Quick Batch (5)")

        row2 = box.row(align=True)
        row2.operator("mesh.test_hair_archetypes", text="All Archetypes")
        row2.operator("mesh.export_hair_samples",  text="Export JSON")

        # Object list
        layout.separator()
        box = layout.box()
        box.label(text="Available Objects:", icon='OBJECT_DATA')
        found = False
        for obj in bpy.data.objects:
            if 'mb_' in obj.name.lower():
                box.label(text=f"  • {obj.name}")
                found = True
        if not found:
            box.label(text="  None found", icon='ERROR')


# =============================================================================
# OPERATORS
# =============================================================================

class MESH_OT_start_bridge(bpy.types.Operator):
    bl_idname = "mesh.start_bridge"
    bl_label  = "Start Bridge"
    def execute(self, context):
        start_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring started")
        return {'FINISHED'}


class MESH_OT_stop_bridge(bpy.types.Operator):
    bl_idname = "mesh.stop_bridge"
    bl_label  = "Stop Bridge"
    def execute(self, context):
        stop_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring stopped")
        return {'FINISHED'}


class MESH_OT_test_male(bpy.types.Operator):
    bl_idname = "mesh.test_male"
    bl_label  = "Test Male"

    def execute(self, context):
        test_data = {
            "properties": {
                "L1_Caucasian": 0.8,
                "L2__Body_Size_max": 0.7,
                "L2__Arms_UpperarmMass-UpperarmTone_max-max": 0.8,
            },
            "gender": "male",
            "prompt": "Test male with short hair",
            "hair_analysis": {
                'has_hair': True, 'style': 'short', 'length': 'short',
                'parameters': {'length': 0.3, 'density': 0.7}
            }
        }
        char = get_object_by_gender("male")
        if char:
            process_enhanced_properties(test_data, char, "male")
            self.report({'INFO'}, "Male test complete")
        else:
            self.report({'ERROR'}, "No male character found")
        return {'FINISHED'}


class MESH_OT_test_female(bpy.types.Operator):
    bl_idname = "mesh.test_female"
    bl_label  = "Test Female"

    def execute(self, context):
        test_data = {
            "properties": {
                "L1_Caucasian": 0.8,
                "L2__Eyes_Size_max": 0.7,
                "L2__Body_Size_min": 0.5,
            },
            "gender": "female",
            "prompt": "Test female with long wavy hair",
            "hair_analysis": {
                'has_hair': True, 'style': 'wavy', 'length': 'long',
                'parameters': {'length': 0.7, 'curl': 0.4, 'density': 0.8}
            }
        }
        char = get_object_by_gender("female")
        if char:
            process_enhanced_properties(test_data, char, "female")
            self.report({'INFO'}, "Female test complete")
        else:
            self.report({'ERROR'}, "No female character found")
        return {'FINISHED'}


class MESH_OT_test_hair(bpy.types.Operator):
    """Test hair generation on the selected object using a single sample"""
    bl_idname = "mesh.test_hair"
    bl_label  = "Test Hair on Selected"

    hair_style: bpy.props.EnumProperty(
        name="Style",
        items=[
            ('straight',   'Straight',   ''),
            ('wavy',       'Wavy',       ''),
            ('curly',      'Curly',      ''),
            ('afro',       'Afro',       ''),
            ('short',      'Short',      ''),
            ('long',       'Long',       ''),
            ('pixie',      'Pixie',      ''),
            ('dreadlocks', 'Dreadlocks', ''),
            ('braided',    'Braided',    ''),
            ('spiky',      'Spiky',      ''),
        ],
        default='wavy'
    )

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}

        # Get an archetype sample matching this style
        if HAIR_SUPPORT:
            samples = sample_generator.get_archetype_samples()
            match   = next(
                (s for s in samples if s.get("style") == self.hair_style), None
            )
            if match:
                test_hair = match
            else:
                test_hair = {
                    'has_hair': True, 'style': self.hair_style,
                    'length': 'medium', 'volume': 'normal',
                    'parameters': {
                        'curl':    0.4 if self.hair_style in ('wavy', 'curly') else 0.0,
                        'density': 0.7,
                        'length':  0.6,
                    }
                }
        else:
            test_hair = {
                'has_hair': True, 'style': self.hair_style,
                'parameters': {'curl': 0.3, 'density': 0.7, 'length': 0.5}
            }

        result = apply_hair_from_analysis(obj, test_hair)
        if result.get('success'):
            self.report({'INFO'}, f"Hair generated: {self.hair_style}")
        else:
            self.report({'WARNING'}, f"Hair issue: {result.get('reason', 'unknown')}")
        return {'FINISHED'}


class MESH_OT_test_hair_batch(bpy.types.Operator):
    """Generate a quick batch of 5 prompt-matched hair samples on selected object"""
    bl_idname = "mesh.test_hair_batch"
    bl_label  = "Quick Hair Batch (5)"

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}

        if not HAIR_SUPPORT:
            self.report({'ERROR'}, "Hair support not available")
            return {'CANCELLED'}

        results = generate_hair_sample_batch(
            character_obj=obj,
            prompt="wavy medium hair",
            count=5,
            strategy="prompt",
            seed=42
        )
        ok = sum(1 for r in results if r.get("success"))
        self.report({'INFO'}, f"Batch done: {ok}/{len(results)} samples ok")
        return {'FINISHED'}


class MESH_OT_test_hair_archetypes(bpy.types.Operator):
    """Cycle through all named archetype samples on selected object"""
    bl_idname = "mesh.test_hair_archetypes"
    bl_label  = "Run All Archetypes"

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}

        if not HAIR_SUPPORT:
            self.report({'ERROR'}, "Hair support not available")
            return {'CANCELLED'}

        results = generate_hair_sample_batch(
            character_obj=obj,
            prompt="",
            count=999,       # All of them
            strategy="archetype"
        )
        ok = sum(1 for r in results if r.get("success"))
        self.report({'INFO'}, f"Archetypes: {ok}/{len(results)} ok")
        return {'FINISHED'}


class MESH_OT_export_hair_samples(bpy.types.Operator):
    """Export the full hair sample library to a JSON file"""
    bl_idname  = "mesh.export_hair_samples"
    bl_label   = "Export Hair Samples JSON"

    def execute(self, context):
        if not HAIR_SUPPORT:
            self.report({'ERROR'}, "Hair support not available")
            return {'CANCELLED'}

        export_path = os.path.join(COMMUNICATION_DIR, "hair_samples_export.json")
        count = hair_analyzer.export_samples(export_path)
        self.report({'INFO'}, f"Exported {count} samples → {export_path}")
        return {'FINISHED'}


# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MESH_PT_character_bridge,
    MESH_OT_start_bridge,
    MESH_OT_stop_bridge,
    MESH_OT_test_male,
    MESH_OT_test_female,
    MESH_OT_test_hair,
    MESH_OT_test_hair_batch,
    MESH_OT_test_hair_archetypes,
    MESH_OT_export_hair_samples,
]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception:
            pass


def unregister():
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


register()

# =============================================================================
# STARTUP BANNER
# =============================================================================
stats = get_hair_sample_stats()

print("\n" + "="*70)
print("🎭 ENHANCED CHARACTER GENERATOR BRIDGE  |  WITH HAIR SAMPLE BATCHING")
print("="*70)
print("📋 Instructions:")
print("  1. Click 'Start Bridge' in Object Properties panel")
print("  2. Open http://127.0.0.1:5000 in your browser")
print("  3. Generate characters, or use hair batch buttons in the panel!")
print()
print("📊 Status:")
print(f"  GBH Tool     : {'✓ AVAILABLE' if GBH_AVAILABLE else '✗ NOT FOUND'}")
print(f"  Hair Analyzer: {'✓ LOADED' if HAIR_SUPPORT else '✗ NOT LOADED'}")
if stats.get("available"):
    print(f"  Samples      : {stats['total_samples']} total  "
          f"({stats['archetypes']} archetypes, "
          f"{stats['color_presets']} colors)")
    print(f"  By source    : {stats.get('by_source', {})}")
print()
print("📁 Communication:")
print(f"  Watch: {COMMUNICATION_DIR}")
print("="*70 + "\n")