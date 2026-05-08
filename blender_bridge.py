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
REQUEST_FILE = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE = os.path.join(COMMUNICATION_DIR, "character_response.json")
HEARTBEAT_FILE = os.path.join(COMMUNICATION_DIR, "blender_heartbeat.json")

# Try to import hair analyzer
try:
    import sys
    # Add the directory where hair_generator.py might be
    possible_paths = [
        os.path.dirname(os.path.dirname(__file__)),
        r"E:\work\work\sem_project\character-generator",
        os.getcwd()
    ]
    for path in possible_paths:
        if path not in sys.path:
            sys.path.append(path)
    
    from hair_generator import hair_analyzer
    HAIR_SUPPORT = True
    print("✓ Hair generator module loaded successfully")
except ImportError as e:
    print(f"⚠️ Hair generator module not available: {e}")
    HAIR_SUPPORT = False
    # Create dummy analyzer for fallback
    class DummyHairAnalyzer:
        def analyze_hair_prompt(self, prompt): 
            return {'has_hair': True, 'style': None, 'length': None, 'color': None}
        def get_gbh_operator_sequence(self, obj, analysis): 
            return []
    hair_analyzer = DummyHairAnalyzer()

# Check for GBH Tool
GBH_AVAILABLE = False
try:
    # Check if GBH addon is enabled
    if hasattr(bpy.context.preferences, 'addons'):
        addon_names = [addon.module for addon in bpy.context.preferences.addons]
        gbh_addons = [name for name in addon_names if 'gbh' in name.lower()]
        if gbh_addons:
            print(f"✓ GBH Tool found: {gbh_addons}")
            GBH_AVAILABLE = True
        else:
            print("⚠️ GBH Tool not found in enabled addons")
    else:
        print("⚠️ Cannot check addons")
except Exception as e:
    print(f"⚠️ GBH Tool check failed: {e}")

is_monitoring = False
last_heartbeat = None
failed_requests = 0
max_failures = 3

# File-based logging so we can read output from outside Blender
BRIDGE_LOG = os.path.join(COMMUNICATION_DIR, "bridge_debug.log")
def log_to_file(msg):
    """Write a message to both console and debug log file"""
    print(msg)
    try:
        with open(BRIDGE_LOG, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except:
        pass

# =============================================================================
# HEARTBEAT FUNCTION
# =============================================================================
def update_heartbeat():
    """Update heartbeat file to show Blender is alive"""
    global last_heartbeat
    try:
        heartbeat = {
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "gbh_available": GBH_AVAILABLE,
            "hair_support": HAIR_SUPPORT,
            "status": "running"
        }
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump(heartbeat, f)
        last_heartbeat = datetime.now()
    except:
        pass
    return 5.0  # Update every 5 seconds

# =============================================================================
# HAIR GENERATION FUNCTIONS (IMPROVED)
# =============================================================================

def find_gbh_operators():
    """Dynamically find all GBH-related operators in Blender"""
    gbh_ops = {
        'generate': [],
        'remove': [],
        'style': [],
        'property': []
    }
    
    try:
        # Scan all operator categories
        for category in dir(bpy.ops):
            if category.startswith('_'):
                continue
            
            try:
                ops_module = getattr(bpy.ops, category)
                for op_name in dir(ops_module):
                    if op_name.startswith('_'):
                        continue
                    
                    full_op = f"{category}.{op_name}"
                    
                    # Check if it's GBH related
                    if 'gbh' in full_op.lower() or 'hair' in full_op.lower():
                        if 'generate' in op_name.lower() or 'create' in op_name.lower() or 'add' in op_name.lower():
                            gbh_ops['generate'].append(full_op)
                        elif 'remove' in op_name.lower() or 'delete' in op_name.lower():
                            gbh_ops['remove'].append(full_op)
                        elif 'style' in op_name.lower() or 'set' in op_name.lower():
                            gbh_ops['style'].append(full_op)
                        else:
                            gbh_ops['property'].append(full_op)
            except:
                continue
    except Exception as e:
        print(f"Error scanning operators: {e}")
    
    return gbh_ops

def generate_hair_with_gbh(character_obj, hair_analysis):
    """
    Generate hair using GBH Tool with dynamic operator detection
    """
    global failed_requests
    
    if not GBH_AVAILABLE:
        print("⚠️ GBH Tool not available - skipping hair generation")
        return {"success": False, "reason": "GBH Tool not installed/enabled"}
    
    if not hair_analysis.get('has_hair', True):
        print("✓ No hair requested (bald/balding detected)")
        return {"success": True, "hair_generated": False, "reason": "bald"}
    
    print(f"\n{'='*60}")
    print(f"💇 GENERATING HAIR WITH GBH TOOL")
    print(f"{'='*60}")
    print(f"Style: {hair_analysis.get('style', 'unknown')}")
    print(f"Length: {hair_analysis.get('length', 'unknown')}")
    print(f"Volume: {hair_analysis.get('volume', 'unknown')}")
    print(f"Color: {hair_analysis.get('color', 'unknown')}")
    print(f"Parameters: {hair_analysis.get('parameters', {})}")
    print(f"{'='*60}\n")
    
    # Find available GBH operators
    gbh_ops = find_gbh_operators()
    
    if not gbh_ops['generate']:
        print("⚠️ No GBH generation operators found. Available GBH operators:")
        for cat, ops in gbh_ops.items():
            if ops:
                print(f"  {cat}: {ops}")
        
        # If no generation operators but we have property operators, try to set hair properties
        if gbh_obs.get('property'):
            print("Attempting to set hair properties on existing hair system...")
            # Try to find existing hair object
            hair_obj = None
            for obj in bpy.data.objects:
                if 'hair' in obj.name.lower():
                    hair_obj = obj
                    break
            
            if hair_obj:
                # Try to set properties
                for op_name in gbh_ops['property']:
                    try:
                        category, op = op_name.split('.')
                        op_func = getattr(getattr(bpy.ops, category), op)
                        
                        # Try with parameters
                        params = {
                            'object': hair_obj.name,
                            'length': hair_analysis.get('parameters', {}).get('length', 0.5),
                            'curl': hair_analysis.get('parameters', {}).get('curl', 0.3),
                            'density': hair_analysis.get('parameters', {}).get('density', 0.7),
                        }
                        
                        # Filter to only accepted parameters (this is tricky without knowing schema)
                        try:
                            op_func(**params)
                            print(f"✓ Set properties using {op_name}")
                            return {"success": True, "hair_generated": True}
                        except TypeError as e:
                            # Parameter mismatch, try without params
                            op_func()
                            print(f"✓ Executed {op_name} (no params)")
                            return {"success": True, "hair_generated": True}
                    except Exception as e:
                        print(f"✗ Failed to execute {op_name}: {e}")
        
        return {"success": False, "reason": "no_generation_operators"}
    
    # Try each generation operator until one works
    for op_name in gbh_ops['generate']:
        try:
            print(f"Trying operator: {op_name}")
            category, op = op_name.split('.')
            op_func = getattr(getattr(bpy.ops, category), op)
            
            # Select the character object first
            bpy.context.view_layer.objects.active = character_obj
            character_obj.select_set(True)
            
            # Try to call with parameters (adjust based on actual GBH API)
            params = {
                'target': character_obj.name,
                'hair_type': 'strands',
                'length': hair_analysis.get('parameters', {}).get('length', 0.5),
                'curl': hair_analysis.get('parameters', {}).get('curl', 0.3),
                'density': hair_analysis.get('parameters', {}).get('density', 0.7),
            }
            
            # Filter out None values
            params = {k: v for k, v in params.items() if v is not None}
            
            try:
                # Try with full parameters
                result = op_func(**params)
                print(f"✓ Successfully executed {op_name} with parameters")
            except TypeError:
                # If parameter mismatch, try without
                result = op_func()
                print(f"✓ Successfully executed {op_name} (no parameters)")
            
            # Reset failure count on success
            failed_requests = 0
            
            return {
                "success": True,
                "hair_generated": True,
                "operator_used": op_name,
                "style": hair_analysis.get('style'),
                "parameters": hair_analysis.get('parameters', {})
            }
            
        except Exception as e:
            print(f"✗ Failed with {op_name}: {e}")
            continue
    
    # If we get here, all operators failed
    failed_requests += 1
    return {"success": False, "reason": "all_operators_failed"}

# =============================================================================
# CLOTHING SYSTEM - Uses CharMorph addon for proper asset fitting
# =============================================================================

# CharMorph asset paths (per gender)
CHARMORPH_ASSETS_DIR = {
    "female": os.path.join(
        os.path.expanduser("~"),
        "AppData", "Roaming", "Blender Foundation", "Blender", "4.5",
        "scripts", "addons", "CharMorph", "data", "characters", "MB-Lab Female", "assets"
    ),
    "male": os.path.join(
        os.path.expanduser("~"),
        "AppData", "Roaming", "Blender Foundation", "Blender", "4.5",
        "scripts", "addons", "CharMorph", "data", "characters", "MB-Lab Male", "assets"
    ),
}

# Map clothing type keywords to CharMorph asset filenames
# These map user-facing types to available .blend assets
CLOTHING_ASSET_MAP = {
    "female": {
        # Tops (CharMorph native assets - these fit MB-Lab bodies)
        "tshirt": "casualsuit01_top",
        "shirt": "casualsuit01_top",
        "top": "casualsuit01_top",
        "tank_top": "RGF.crop.top.shirt",
        "crop_top": "RGF.crop.top.shirt",
        "jacket": "RGF.Diversion.jumper",
        "jumper": "RGF.Diversion.jumper",
        "sweater": "RGF.Diversion.jumper",
        # Bottoms
        "pants": "trousers",
        "trousers": "trousers",
        "dress_pants": "trousers_01",
        "skirt": "elegantsuit01_btm",
        "shorts": "casualsuit01_btm",
        "jeans": "jeans",
        "cargo_pants": "RGF.pants",
        "sport_pants": "sportsuit01_btm",
        "motorcycle_pants": "Motorcyclepants",
        "casual_pants": "casualsuit02_btm",
        # Dresses
        "dress": "elegantsuit01_btm",
        # Footwear
        "boots": "tactical_boots",
        "ankle_boots": "RGF.Platform.ankle.boots",
        "platform_boots": "RGF.Platform.ankle.boots",
        "shoes": "RGF.DNA.boots",
        # Accessories
        "gloves": "RGF.Diversion.gloves",
        "thong": "RGF.thongs",
        "underwear": "RGF.thongs",
    },
    "male": {
        "tshirt": "tactical_top",
        "shirt": "tactical_top",
        "top": "tactical_top",
        "tank_top": "tactical_top",
        "jacket": "tactical_top",
        "sweater": "tactical_top",
        "pants": "tactical_btm",
        "trousers": "tactical_btm",
        "shorts": "tactical_btm",
        "jeans": "tactical_btm",
        "boots": "tactical_boots",
        "shoes": "tactical_boots",
    },
}

CLOTHING_COLORS = {
    "white": (1.0, 1.0, 1.0, 1.0),
    "black": (0.02, 0.02, 0.02, 1.0),
    "red": (0.8, 0.1, 0.1, 1.0),
    "blue": (0.1, 0.2, 0.8, 1.0),
    "green": (0.1, 0.6, 0.2, 1.0),
    "gray": (0.4, 0.4, 0.4, 1.0),
    "brown": (0.4, 0.25, 0.1, 1.0),
    "navy": (0.05, 0.1, 0.3, 1.0),
    "beige": (0.76, 0.7, 0.5, 1.0),
    "pink": (0.9, 0.5, 0.6, 1.0),
    "yellow": (0.9, 0.85, 0.2, 1.0),
    "purple": (0.5, 0.1, 0.6, 1.0),
    "orange": (0.9, 0.5, 0.1, 1.0),
}


def set_clothing_color(obj, color_name):
    """Apply a color material to a clothing object"""
    rgba = CLOTHING_COLORS.get(color_name, CLOTHING_COLORS["white"])
    mat_name = f"Clothing_{obj.name}_{color_name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.7
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def find_asset_blend(gender, clothing_type):
    """Find the .blend file for a clothing asset"""
    gender_key = gender.lower() if gender.lower() in CLOTHING_ASSET_MAP else "female"
    asset_map = CLOTHING_ASSET_MAP.get(gender_key, {})
    asset_name = asset_map.get(clothing_type)

    if not asset_name:
        log_to_file(f"    No asset mapping for '{clothing_type}' ({gender_key})")
        return None, None

    assets_dir = CHARMORPH_ASSETS_DIR.get(gender_key, "")

    # Check for direct .blend file
    blend_path = os.path.join(assets_dir, f"{asset_name}.blend")
    if os.path.exists(blend_path):
        return blend_path, asset_name

    # Check for subdirectory with asset.blend
    subdir_path = os.path.join(assets_dir, asset_name, "asset.blend")
    if os.path.exists(subdir_path):
        return subdir_path, asset_name

    log_to_file(f"    Asset file not found: {blend_path}")
    return None, None


def import_clothing_from_blend(blend_path, asset_name):
    """Import a clothing mesh from a .blend file"""
    log_to_file(f"    Importing from: {blend_path}")

    # List objects in the blend file
    with bpy.data.libraries.load(blend_path) as (data_from, data_to):
        # Import all objects from the file
        data_to.objects = data_from.objects[:]
        log_to_file(f"    Found objects: {data_from.objects[:]}")

    # Link imported objects to the scene
    imported = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            imported.append(obj)
            log_to_file(f"    Linked: {obj.name} (type={obj.type})")

    # Return the first mesh object
    for obj in imported:
        if obj.type == 'MESH':
            return obj

    return None


def fit_with_charmorph(character_obj, asset_obj):
    """Use CharMorph's fitting system to fit an asset to the character"""
    try:
        ui = bpy.context.window_manager.charmorph_ui
        ui.fitting_char = character_obj
        ui.fitting_asset = asset_obj
        ui.fitting_transforms = True
        result = bpy.ops.charmorph.fit_local()
        log_to_file(f"    CharMorph fit result: {result}")
        return result == {'FINISHED'}
    except Exception as e:
        log_to_file(f"    CharMorph fit failed: {e}")
        return False


def generate_clothing_piece(character_obj, clothing_type, color="white", gender="female"):
    """
    Generate a clothing piece using CharMorph fitting with original CharMorph assets.
    """
    log_to_file(f"  Creating {clothing_type} ({color}) for {gender}...")

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Find the asset blend file
    blend_path, asset_name = find_asset_blend(gender, clothing_type)
    if not blend_path:
        log_to_file(f"  No asset found for {clothing_type}")
        return None

    # Import the clothing mesh
    asset_obj = import_clothing_from_blend(blend_path, asset_name)
    if not asset_obj:
        log_to_file(f"  Failed to import clothing mesh")
        return None

    log_to_file(f"    Imported clothing: {asset_obj.name}")

    # Fit using CharMorph (works with CharMorph-native assets)
    fitted = fit_with_charmorph(character_obj, asset_obj)

    if not fitted:
        log_to_file(f"    WARNING: CharMorph fitting failed")

    # Apply color
    set_clothing_color(asset_obj, color)

    log_to_file(f"  DONE: {clothing_type} fitted and colored ({color})")
    return asset_obj


def remove_existing_clothing():
    """Remove any previously generated/fitted clothing objects"""
    to_remove = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'charmorph_fit_id' in obj.data:
            to_remove.append(obj)
        elif obj.name.startswith("Clothing_"):
            to_remove.append(obj)

    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    if to_remove:
        log_to_file(f"  Removed {len(to_remove)} existing clothing objects")


def apply_clothing_from_analysis(character_obj, clothing_data, gender="female"):
    """
    Main entry point for clothing generation using CharMorph fitting.
    """
    if not clothing_data or not clothing_data.get("items"):
        log_to_file("No clothing requested")
        return {"success": True, "clothing_generated": False}

    log_to_file(f"\n{'='*60}")
    log_to_file(f"GENERATING CLOTHING ({len(clothing_data['items'])} items)")
    log_to_file(f"{'='*60}")

    # Remove old clothing
    remove_existing_clothing()

    results = []
    for item in clothing_data["items"]:
        clothing_type = item.get("type", "").lower()
        color = item.get("color", "white").lower()

        try:
            obj = generate_clothing_piece(character_obj, clothing_type, color, gender)
            if obj:
                results.append({
                    "type": clothing_type,
                    "color": color,
                    "object": obj.name,
                    "success": True
                })
            else:
                results.append({
                    "type": clothing_type,
                    "success": False,
                    "reason": "asset_not_found"
                })
        except Exception as e:
            log_to_file(f"  Error creating {clothing_type}: {e}")
            traceback.print_exc()
            results.append({
                "type": clothing_type,
                "success": False,
                "reason": str(e)
            })

    bpy.context.view_layer.update()

    success_count = sum(1 for r in results if r["success"])
    log_to_file(f"Clothing complete: {success_count}/{len(results)} items")

    return {
        "success": success_count > 0,
        "clothing_generated": True,
        "items": results
    }


def apply_hair_from_analysis(character_obj, hair_analysis):
    """Wrapper for hair generation with error handling"""
    try:
        return generate_hair_with_gbh(character_obj, hair_analysis)
    except Exception as e:
        print(f"✗ Hair generation error: {e}")
        traceback.print_exc()
        return {"success": False, "reason": str(e), "hair_generated": False}

# =============================================================================
# BLENDER HELPER FUNCTIONS (ENHANCED)
# =============================================================================

def get_object_by_gender(gender="male"):
    """Get the appropriate character object with better fallback"""
    # Try exact matches first
    exact_names = [f"mb_{gender}", f"MB_{gender}", f"Mb_{gender}"]
    
    for name in exact_names:
        if name in bpy.data.objects:
            obj = bpy.data.objects[name]
            print(f"✓ Found exact match: {obj.name}")
            return obj
    
    # Try case-insensitive search
    gender_lower = gender.lower()
    for obj in bpy.data.objects:
        obj_lower = obj.name.lower()
        if f"mb_{gender_lower}" in obj_lower:
            print(f"✓ Found character object: {obj.name}")
            return obj
    
    # Try the opposite gender as fallback
    fallback = "female" if gender == "male" else "male"
    for obj in bpy.data.objects:
        obj_lower = obj.name.lower()
        if f"mb_{fallback}" in obj_lower:
            print(f"⚠️ Using fallback {fallback} object: {obj.name}")
            return obj
    
    # Last resort: any object with 'mb_' in name
    for obj in bpy.data.objects:
        if 'mb_' in obj.name.lower():
            print(f"⚠️ Using generic MB object: {obj.name}")
            return obj
    
    print(f"✗ Could not find any character object for gender: {gender}")
    return None

def reset_character_shape_keys(obj):
    """Reset all shape keys with validation"""
    if not obj:
        print("✗ Cannot reset shape keys: No object provided")
        return
    
    if not hasattr(obj.data, "shape_keys") or not obj.data.shape_keys:
        print("⚠️ Object has no shape keys")
        return
    
    try:
        print("--- Resetting all shape keys ---")
        reset_count = 0
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name != "Basis":
                kb.value = 0.0
                reset_count += 1
        print(f"✓ Reset {reset_count} shape keys")
    except Exception as e:
        print(f"✗ Error resetting shape keys: {e}")

def apply_morph(obj, shape_key_name, value):
    """Safely apply morph with validation"""
    if not obj:
        return False
    
    if not hasattr(obj.data, "shape_keys") or not obj.data.shape_keys:
        return False
    
    try:
        if shape_key_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[shape_key_name].value = min(1.0, max(0.0, float(value)))
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
        "morphs_failed": 0,
        "morph_details": [],
        "hair_result": {"success": False, "hair_generated": False},
        "clothing_result": {"success": False, "clothing_generated": False},
        "errors": []
    }
    
    try:
        # Step 1: Reset and apply morphs
        print(f"\n--- Starting Morph Application for {gender.upper()} ---")
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
        print(f"✓ Applied {results['morphs_applied']} morphs, {results['morphs_failed']} failed")
        
        # Step 2: Generate hair
        original_prompt = structured_data.get("prompt", "")
        hair_analysis = structured_data.get("hair_analysis", {})
        
        # If hair_analysis not in structured_data, analyze from prompt
        if not hair_analysis and original_prompt and HAIR_SUPPORT:
            print("\n--- Analyzing Hair from Prompt ---")
            hair_analysis = hair_analyzer.analyze_hair_prompt(original_prompt)
            print(f"Analysis: {hair_analysis}")
        
        if hair_analysis:
            print("\n--- Generating Hair ---")
            hair_result = apply_hair_from_analysis(character_obj, hair_analysis)
            results["hair_result"] = hair_result
            
            if hair_result.get("success"):
                print(f"✓ Hair generation successful: {hair_result.get('style', 'unknown')}")
            else:
                print(f"⚠️ Hair generation issue: {hair_result.get('reason', 'unknown')}")
        else:
            print("\n--- No hair analysis available ---")

        # Step 3: Generate clothing
        clothing_data = structured_data.get("clothing", {})
        log_to_file(f"Clothing data received: {clothing_data}")
        if clothing_data and clothing_data.get("items"):
            log_to_file(f"Generating Clothing ({len(clothing_data['items'])} items)")
            try:
                clothing_result = apply_clothing_from_analysis(character_obj, clothing_data, gender)
                results["clothing_result"] = clothing_result

                if clothing_result.get("success"):
                    items_created = sum(1 for i in clothing_result.get("items", []) if i.get("success"))
                    log_to_file(f"Clothing generation successful: {items_created} items")
                else:
                    log_to_file(f"Clothing generation failed: {clothing_result}")
            except Exception as clothing_err:
                log_to_file(f"CLOTHING ERROR: {clothing_err}")
                log_to_file(traceback.format_exc())
                results["errors"].append(f"Clothing error: {clothing_err}")
        else:
            log_to_file("No clothing requested")

    except Exception as e:
        error_msg = f"Error in process_enhanced_properties: {e}"
        print(f"✗ {error_msg}")
        traceback.print_exc()
        results["errors"].append(error_msg)
    
    return results

# =============================================================================
# BRIDGE MONITORING FUNCTIONS (ENHANCED)
# =============================================================================

def start_bridge_monitoring():
    """Start monitoring with heartbeat"""
    global is_monitoring, failed_requests
    
    if is_monitoring:
        print("Bridge monitoring is already active.")
        return
    
    os.makedirs(COMMUNICATION_DIR, exist_ok=True)
    
    # Reset failure counter
    failed_requests = 0
    
    # Clean up stale files
    for f in [REQUEST_FILE, RESPONSE_FILE]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass
    
    print(f"\n{'='*70}")
    print(f"🎭 STARTING ENHANCED BLENDER BRIDGE WITH HAIR GENERATION")
    print(f"{'='*70}")
    print(f"📁 Watch directory: {COMMUNICATION_DIR}")
    print(f"💇 GBH Tool available: {GBH_AVAILABLE}")
    print(f"🧬 Hair analyzer: {'✓ Loaded' if HAIR_SUPPORT else '✗ Not available'}")
    print(f"⚡ Max failures before auto-stop: {max_failures}")
    print(f"{'='*70}\n")
    
    # Find and display GBH operators
    if GBH_AVAILABLE:
        gbh_ops = find_gbh_operators()
        print("📋 Available GBH operators:")
        for category, ops in gbh_ops.items():
            if ops:
                print(f"  {category}: {', '.join(ops[:3])}...")
    
    is_monitoring = True
    
    # Register timers
    if not bpy.app.timers.is_registered(check_for_requests):
        bpy.app.timers.register(check_for_requests, first_interval=1.0)
    
    if not bpy.app.timers.is_registered(update_heartbeat):
        bpy.app.timers.register(update_heartbeat, first_interval=1.0)
    
    print("✅ Bridge monitoring started")

def stop_bridge_monitoring():
    """Stop monitoring and cleanup"""
    global is_monitoring
    
    is_monitoring = False
    
    if bpy.app.timers.is_registered(check_for_requests):
        bpy.app.timers.unregister(check_for_requests)
    
    if bpy.app.timers.is_registered(update_heartbeat):
        bpy.app.timers.unregister(update_heartbeat)
    
    # Write final heartbeat
    try:
        heartbeat = {
            "timestamp": datetime.now().isoformat(),
            "status": "stopped"
        }
        with open(HEARTBEAT_FILE, 'w') as f:
            json.dump(heartbeat, f)
    except:
        pass
    
    print("Bridge monitoring stopped.")

def check_for_requests():
    """Enhanced request checker with failure handling"""
    global is_monitoring, failed_requests
    
    if not is_monitoring:
        return None
    
    # Auto-stop if too many failures
    if failed_requests >= max_failures:
        print(f"\n⚠️ Too many failures ({failed_requests}). Auto-stopping bridge.")
        print("Click 'Start Bridge' in Blender UI to restart.")
        stop_bridge_monitoring()
        return None
    
    try:
        if os.path.exists(REQUEST_FILE):
            # Read and process request
            with open(REQUEST_FILE, 'r', encoding='utf-8') as f:
                request_data = json.load(f)
            
            # Extract data
            structured_data = request_data.get('structured_data', {})
            prompt = request_data.get('prompt', '')
            
            if not structured_data:
                print("✗ Error: Missing structured_data")
                send_error_response(prompt, "Missing structured_data")
                os.remove(REQUEST_FILE)
                failed_requests += 1
                return 0.5
            
            # Get gender and find character
            gender = structured_data.get('gender', 'male')
            print(f"\n{'='*60}")
            print(f"🎭 PROCESSING REQUEST: {gender.upper()}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"{'='*60}")
            
            character = get_object_by_gender(gender)
            
            if character:
                # Process the request
                results = process_enhanced_properties(structured_data, character, gender)
                
                # Send success response
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "status": "completed",
                    "message": f"✓ {gender.capitalize()} character generated",
                    "morphs_applied": results["morphs_applied"],
                    "morphs_failed": results["morphs_failed"],
                    "hair_generated": results["hair_result"].get("hair_generated", False),
                    "hair_style": results["hair_result"].get("style", "unknown"),
                    "hair_success": results["hair_result"].get("success", False),
                    "character_object": character.name,
                    "errors": results["errors"]
                }
                
                with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(response_data, f, indent=2)
                
                print(f"\n✅ Response sent - Morphs: {results['morphs_applied']}, Hair: {results['hair_result'].get('hair_generated', False)}")
                
                # Reset failure count on success
                failed_requests = 0
            else:
                # Character not found
                error_msg = f"Could not find {gender} character object"
                print(f"✗ {error_msg}")
                send_error_response(prompt, error_msg, gender)
                failed_requests += 1
            
            # Clean up request file
            try:
                os.remove(REQUEST_FILE)
            except:
                pass
    
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in request file: {e}")
        failed_requests += 1
        try:
            os.remove(REQUEST_FILE)
        except:
            pass
    
    except Exception as e:
        print(f"✗ Error processing request: {e}")
        traceback.print_exc()
        failed_requests += 1
        
        try:
            send_error_response("Unknown", f"Error: {str(e)}")
            if os.path.exists(REQUEST_FILE):
                os.remove(REQUEST_FILE)
        except:
            pass
    
    return 0.5

def send_error_response(prompt, error_message, gender="unknown"):
    """Helper to send error responses"""
    try:
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "gender": gender,
            "status": "error",
            "message": error_message,
            "morphs_applied": 0,
            "hair_generated": False
        }
        with open(RESPONSE_FILE, 'w', encoding='utf-8') as f:
            json.dump(error_response, f, indent=2)
    except:
        pass

# =============================================================================
# BLENDER UI PANEL (UPDATED)
# =============================================================================

class MESH_PT_character_bridge(bpy.types.Panel):
    """Enhanced UI Panel"""
    bl_label = "Character Generator Bridge"
    bl_idname = "MESH_PT_character_bridge"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    
    def draw(self, context):
        layout = self.layout
        
        # Status header
        box = layout.box()
        box.label(text="BRIDGE STATUS", icon='INFO')
        
        if is_monitoring:
            box.label(text="● RUNNING", icon='CHECKMARK')
        else:
            box.label(text="○ STOPPED", icon='X')
        
        # GBH Status
        if GBH_AVAILABLE:
            box.label(text="✓ GBH Tool: Available", icon='CHECKMARK')
        else:
            box.label(text="✗ GBH Tool: Not Found", icon='ERROR')
        
        if HAIR_SUPPORT:
            box.label(text="✓ Hair Analyzer: Loaded", icon='CHECKMARK')
        else:
            box.label(text="✗ Hair Analyzer: Not Loaded", icon='ERROR')
        
        # Control buttons
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        
        if is_monitoring:
            row.operator("mesh.stop_bridge", text="Stop Bridge", icon='PAUSE')
        else:
            row.operator("mesh.start_bridge", text="Start Bridge", icon='PLAY')
        
        # Test buttons
        layout.separator()
        box = layout.box()
        box.label(text="TEST GENERATION", icon='EXPERIMENTAL')
        
        row = box.row(align=True)
        row.operator("mesh.test_male", text="Test Male")
        row.operator("mesh.test_female", text="Test Female")
        
        row = box.row(align=True)
        row.operator("mesh.test_hair", text="Test Hair on Selected")
        
        # Quick info
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

class MESH_OT_start_bridge(bpy.types.Operator):
    """Start the bridge monitoring"""
    bl_idname = "mesh.start_bridge"
    bl_label = "Start Bridge"
    
    def execute(self, context):
        start_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring started")
        return {'FINISHED'}

class MESH_OT_stop_bridge(bpy.types.Operator):
    """Stop the bridge monitoring"""
    bl_idname = "mesh.stop_bridge"
    bl_label = "Stop Bridge"
    
    def execute(self, context):
        stop_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring stopped")
        return {'FINISHED'}

class MESH_OT_test_male(bpy.types.Operator):
    """Test male character generation"""
    bl_idname = "mesh.test_male"
    bl_label = "Test Male"
    
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
                'has_hair': True,
                'style': 'short',
                'length': 'short',
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
    """Test female character generation"""
    bl_idname = "mesh.test_female"
    bl_label = "Test Female"
    
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
                'has_hair': True,
                'style': 'wavy',
                'length': 'long',
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
    """Test hair generation on selected object"""
    bl_idname = "mesh.test_hair"
    bl_label = "Test Hair on Selected"
    
    hair_style: bpy.props.EnumProperty(
        name="Style",
        items=[
            ('straight', 'Straight', ''),
            ('wavy', 'Wavy', ''),
            ('curly', 'Curly', ''),
            ('short', 'Short', ''),
            ('long', 'Long', ''),
            ('pixie', 'Pixie', ''),
        ],
        default='wavy'
    )
    
    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "No object selected")
            return {'CANCELLED'}
        
        test_hair = {
            'has_hair': True,
            'style': self.hair_style,
            'length': 'medium',
            'volume': 'normal',
            'parameters': {
                'curl': 0.4 if self.hair_style in ['wavy', 'curly'] else 0.0,
                'density': 0.7,
                'length': 0.6 if self.hair_style == 'long' else 0.3,
            }
        }
        
        result = apply_hair_from_analysis(obj, test_hair)
        
        if result.get('success'):
            self.report({'INFO'}, f"Hair generated: {self.hair_style}")
        else:
            self.report({'WARNING'}, f"Hair generation issue: {result.get('reason', 'unknown')}")
        
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
]

def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except:
            pass

def unregister():
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass

# Auto-register
register()

# =============================================================================
# STARTUP BANNER
# =============================================================================
print("\n" + "="*70)
print("🎭 ENHANCED CHARACTER GENERATOR BRIDGE WITH HAIR")
print("="*70)
print("📋 Instructions:")
print("  1. Click 'Start Bridge' in the Object Properties panel")
print("  2. Open http://127.0.0.1:5000 in your browser")
print("  3. Generate characters with hair descriptions!")
print()
print("📊 Status:")
print(f"  GBH Tool: {'✓ AVAILABLE' if GBH_AVAILABLE else '✗ NOT FOUND'}")
print(f"  Hair Analyzer: {'✓ LOADED' if HAIR_SUPPORT else '✗ NOT LOADED'}")
print()
print("📁 Communication:")
print(f"  Watch: {COMMUNICATION_DIR}")
print("="*70 + "\n")

# Auto-start? Uncomment if desired
# start_bridge_monitoring()