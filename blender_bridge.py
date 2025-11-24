import bpy
from mathutils import Vector
import re
import json
import os
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import traceback
import shutil
import subprocess
import uuid


# ---------- Path to the uploaded reference image (from your session) ----------
REFERENCE_IMAGE_PATH = r"/mnt/data/d4f73a0c-9148-4e7b-a5d1-d0e67cac2171.png"

# ---------- HairNet integration configuration ----------
# Point this to the local hairnet-ai repo (adjust to your install path)
HAIRNET_REPO_PATH = r"C:\path\to\hairnet-ai"  # <<-- CHANGE THIS to your repo path
# Template command used to run hairnet. Replace flags with the actual script your installation uses.
# The placeholders will be filled in by run_hairnet_pipeline()
HAIRNET_CMD_TEMPLATE = (
    r"python {repo}/run_hairnet.py "
    r"--head {head_obj} "
    r"--init_mesh {hair_obj} "
    r"--scalp {scalp_obj} "
    r"--outdir {outdir} "
    r"--style '{style}' "
)

# =============================================================================
# BLENDER BRIDGE CONFIGURATION
# =============================================================================
COMMUNICATION_DIR = r"C:\temp\blender_bridge"
REQUEST_FILE = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE = os.path.join(COMMUNICATION_DIR, "character_response.json")
STATUS_FILE = os.path.join(COMMUNICATION_DIR, "blender_status.json")
LOG_FILE = os.path.join(COMMUNICATION_DIR, "blender_log.txt")

is_monitoring = False

# =============================================================================
# LOGGING SYSTEM
# =============================================================================
def log(message, level="INFO"):
    """Write to both console and log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {message}"
    print(log_message)
    
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    except:
        pass

def update_status(status="active"):
    """Update status file so frontend knows bridge is running"""
    try:
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "monitoring": is_monitoring
        }
        with open(STATUS_FILE, 'w') as f:
            json.dump(status_data, f)
    except Exception as e:
        log(f"Failed to update status: {e}", "ERROR")

# =============================================================================
# BLENDER HELPER FUNCTIONS
# =============================================================================
def get_object_by_gender(gender="male"):
    """Get the appropriate character object based on detected gender."""
    log(f"Searching for {gender} character object...")
    
    # Try exact name first
    base_name = "mb_female" if gender == "female" else "mb_male"
    
    if base_name in bpy.data.objects:
        log(f"Found exact match: {base_name}")
        return bpy.data.objects[base_name]
    
    # Try with prefix
    for obj in bpy.data.objects:
        if obj.name.startswith(base_name):
            log(f"Found with prefix: {obj.name}")
            return obj
    
    # Try fallback gender
    fallback_name = "mb_male" if gender == "female" else "mb_female"
    log(f"Trying fallback: {fallback_name}")
    
    if fallback_name in bpy.data.objects:
        log(f"Found fallback: {fallback_name}", "WARNING")
        return bpy.data.objects[fallback_name]
    
    for obj in bpy.data.objects:
        if obj.name.startswith(fallback_name):
            log(f"Found fallback with prefix: {obj.name}", "WARNING")
            return obj
    
    # Last resort: find any mesh object that looks like a character
    log("Searching for any character-like mesh...", "WARNING")
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data.shape_keys and len(obj.data.vertices) > 1000:
            log(f"Using generic mesh as character: {obj.name}", "WARNING")
            return obj
    
    log("No character object found!", "ERROR")
    return None

def reset_character_shape_keys(obj):
    """Resets all shape key values to 0.0 for a clean start."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        log("No shape keys to reset")
        return
    
    log("Resetting all shape keys...")
    reset_count = 0
    for kb in obj.data.shape_keys.key_blocks:
        if kb.name != "Basis":
            kb.value = 0.0
            reset_count += 1
    log(f"Reset {reset_count} shape keys")

def apply_morph(obj, shape_key_name, value):
    """Applies a single morph value, checking if the key exists."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        return False
    
    if shape_key_name in obj.data.shape_keys.key_blocks:
        obj.data.shape_keys.key_blocks[shape_key_name].value = min(1.0, max(0.0, value))
        return True
    return False

# =============================================================================
# BODY PROPERTY PROCESSING (STEP 1)
# =============================================================================
def process_enhanced_properties(structured_data: Dict, character_obj, gender: str):
    """STEP 1: Applies body morphs from enhanced property mapping"""
    log(f"\n{'='*60}")
    log(f"STEP 1: APPLYING BODY PROPERTIES")
    log(f"{'='*60}")
    log(f"Processing {gender.upper()} character: {character_obj.name}")
    
    reset_character_shape_keys(character_obj)
    
    properties = structured_data.get("properties", {})
    
    if not properties:
        log("WARNING: No properties to apply!", "WARNING")
        return 0, 0
    
    log(f"Total properties to apply: {len(properties)}")
    
    applied_count = 0
    failed_count = 0
    failed_properties = []
    
    for property_name, intensity in properties.items():
        if apply_morph(character_obj, property_name, intensity):
            applied_count += 1
            if applied_count <= 5:  # Log first 5 for verification
                log(f"  ✓ Applied: {property_name} = {intensity:.2f}")
        else:
            failed_count += 1
            failed_properties.append(property_name)
    
    # Force viewport update
    bpy.context.view_layer.update()
    
    log(f"✓ Body properties applied: {applied_count}")
    log(f"✗ Body properties failed: {failed_count}")
    
    if failed_count > 0 and failed_count <= 10:
        log(f"Failed properties: {', '.join(failed_properties[:10])}", "WARNING")
    
    return applied_count, failed_count

# =============================================================================
# HAIR GENERATION (STEP 2) - SIMPLIFIED VERSION
# =============================================================================
def create_head_vertex_group_safe(character_obj):
    """Create vertex group for head region - FIXED VERSION"""
    try:
        mesh = character_obj.data
        
        # Remove existing hair vertex group if it exists
        if "Hair_Region" in character_obj.vertex_groups:
            character_obj.vertex_groups.remove(character_obj.vertex_groups["Hair_Region"])
        
        # Create new vertex group
        hair_group = character_obj.vertex_groups.new(name="Hair_Region")
        
        # Get world coordinates and find head region
        vertices = mesh.vertices
        world_matrix = character_obj.matrix_world
        
        # Find character bounds to determine head region
        z_coords = [(i, (world_matrix @ v.co).z) for i, v in enumerate(vertices)]
        if not z_coords:
            return None
            
        z_coords.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate head region more precisely
        max_z = z_coords[0][1]
        min_z = z_coords[-1][1]
        total_height = max_z - min_z
        
        # Head is approximately top 15% of character height
        head_threshold = max_z - (total_height * 0.15)
        
        # Also consider X and Y coordinates to exclude non-head areas
        head_vertex_indices = []
        
        for i, vertex in enumerate(vertices):
            world_coord = world_matrix @ vertex.co
            
            # Select vertices in top 15% by height AND within head area
            if world_coord.z >= head_threshold:
                # Calculate distance from head center (approximate)
                # Head is roughly centered and narrower than body
                head_center_x = 0  # Assuming character is centered on X axis
                head_center_y = 0  # Assuming character is centered on Y axis
                
                distance_x = abs(world_coord.x - head_center_x)
                distance_y = abs(world_coord.y - head_center_y)
                
                # Head is roughly within these bounds (adjust based on your character)
                max_head_width = total_height * 0.2
                max_head_depth = total_height * 0.15
                
                if distance_x < max_head_width and distance_y < max_head_depth:
                    head_vertex_indices.append(i)
        
        # If we didn't get enough vertices, use top vertices by Z coordinate
        if len(head_vertex_indices) < 50:
            head_vertex_indices = []
            # Take top 10% of vertices by Z coordinate
            num_head_verts = max(100, int(len(vertices) * 0.1))
            for i, (vertex_idx, z_val) in enumerate(z_coords[:num_head_verts]):
                head_vertex_indices.append(vertex_idx)
        
        # Add vertices to group with weight 1.0
        if head_vertex_indices:
            hair_group.add(head_vertex_indices, 1.0, 'REPLACE')
            log(f"✓ Created Hair_Region with {len(head_vertex_indices)} vertices")
            
            # DEBUG: Print some vertex positions to verify
            if len(head_vertex_indices) > 0:
                sample_idx = head_vertex_indices[0]
                sample_vertex = vertices[sample_idx]
                world_pos = world_matrix @ sample_vertex.co
                log(f"✓ Sample head vertex at Z: {world_pos.z:.3f} (threshold: {head_threshold:.3f})")
            
            return "Hair_Region"
        else:
            log("⚠ No vertices found for head region", "WARNING")
            return None
            
    except Exception as e:
        log(f"⚠ Vertex group creation failed: {e}", "WARNING")
        import traceback
        log(traceback.format_exc(), "ERROR")
        return None

def create_simple_hair_system(character_obj, hair_params):
    """Create particle hair system - FIXED HEAD CONTAINMENT"""
    try:
        log(f"\n{'='*60}")
        log(f"STEP 2: CREATING HAIR SYSTEM")
        log(f"{'='*60}")
        
        # Make character active
        bpy.context.view_layer.objects.active = character_obj
        character_obj.select_set(True)
        log(f"Set active object: {character_obj.name}")
        
        # Remove any existing hair systems
        existing_systems = [psys.name for psys in character_obj.particle_systems]
        for psys_name in existing_systems:
            character_obj.particle_systems.active_index = 0
            bpy.ops.object.particle_system_remove()
        log(f"Cleared {len(existing_systems)} existing particle systems")
        
        # Add particle system
        log("Adding particle system...")
        bpy.ops.object.particle_system_add()
        
        psys = character_obj.particle_systems[-1]
        psys.name = "Hair_System"
        
        settings = psys.settings
        settings.type = 'HAIR'
        log("✓ Set type to HAIR")
        
        # Extract parameters
        particle_count = hair_params.get('particle_count', 5000)
        particle_length = hair_params.get('particle_length', 0.5)
        curl_intensity = hair_params.get('curl_intensity', 0.0)
        randomness = hair_params.get('randomness', 0.3)
        
        # Configure hair parameters
        settings.count = int(particle_count)
        settings.hair_length = float(particle_length)
        settings.hair_step = 5
        settings.render_step = 5
        
        log(f"✓ Configured: {settings.count} particles, length {settings.hair_length}")
        
        # Add children for volume but fewer to reduce body coverage
        settings.child_nbr = max(10, int(settings.count * 0.02))  # REDUCED to 2%
        settings.child_length = 0.5  # REDUCED from 0.8
        settings.child_radius = float(randomness) * 0.3  # REDUCED randomness
        
        log(f"✓ Added child particles: {settings.child_nbr}")
        
        # Set curl if needed
        if curl_intensity > 0:
            settings.child_type = 'INTERPOLATED'
            settings.clump_factor = 0.1 + curl_intensity * 0.2
            settings.roughness_1 = curl_intensity * 0.8
            log(f"✓ Set curl intensity: {curl_intensity}")
        
        # CRITICAL: Limit to head region
        vertex_group = create_head_vertex_group_safe(character_obj)
        if vertex_group:
            settings.vertex_group_density = vertex_group
            # Also limit children to the same region
            settings.vertex_group_clump = vertex_group  
            settings.vertex_group_length = vertex_group
            settings.vertex_group_kink = vertex_group
            settings.vertex_group_roughness = vertex_group
            
            log(f"✓ LIMITED HAIR TO HEAD REGION: {vertex_group}")
            
            # Set density to 0 outside vertex group
            settings.density_factor = 1.0
            settings.use_density_factor = True
        else:
            log("❌ WARNING: No vertex group created - hair will be on full body!", "WARNING")
        
        # Additional settings to contain hair
        settings.use_even_distribution = True
        settings.use_modifier_stack = True
        settings.use_hair_dynamics = False  # Disable physics for stability
        
        # Set emission to only from vertices (not faces)
        settings.emit_from = 'VERT'
        settings.use_emit_random = True
        
        # Force update and viewport refresh
        bpy.context.view_layer.update()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        time.sleep(1.0)  # Give Blender more time to process
        
        log(f"\n✅ HAIR SYSTEM CREATED SUCCESSFULLY!")
        log(f"   System name: {psys.name}")
        log(f"   Particle count: {settings.count}")
        log(f"   Hair length: {settings.hair_length}")
        log(f"   Limited to head: {'YES' if vertex_group else 'NO'}")
        log(f"   Child particles: {settings.child_nbr}")
        
        return True
        
    except Exception as e:
        log(f"✗ Hair creation failed: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return False

# =============================================================================
# BRIDGE MONITORING
# =============================================================================
def start_bridge_monitoring():
    """Start monitoring for character generation requests."""
    global is_monitoring
   
    if is_monitoring:
        log("Bridge monitoring is already active.")
        return
   
    os.makedirs(COMMUNICATION_DIR, exist_ok=True)
   
    log(f"{'='*70}")
    log(f"🎭 BLENDER BRIDGE STARTED")
    log(f"{'='*70}")
    log(f"📁 Watching: {COMMUNICATION_DIR}")
    log(f"📝 Log file: {LOG_FILE}")
    log("⏳ Ready for requests...")
   
    is_monitoring = True
    update_status("active")
    bpy.app.timers.register(check_for_requests, first_interval=0.5)

def stop_bridge_monitoring():
    """Stop monitoring for requests."""
    global is_monitoring
    is_monitoring = False
   
    if bpy.app.timers.is_registered(check_for_requests):
        bpy.app.timers.unregister(check_for_requests)
   
    update_status("stopped")
    log("Bridge monitoring stopped.")


# Utility: create a unique temp folder for each run
def _make_temp_dir(prefix="hairnet_run"):
    tmp = os.path.join(COMMUNICATION_DIR, f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    os.makedirs(tmp, exist_ok=True)
    return tmp

# Export a selected object to OBJ file (with modifiers applied if requested)
def export_object_as_obj(obj, filepath, apply_modifiers=False):
    # Deselect all, select provided, export
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    export_kwargs = {
        "filepath": filepath,
        "use_selection": True,
        "use_mesh_modifiers": apply_modifiers,
        "axis_forward": "-Z",
        "axis_up": "Y",
        "use_triangles": False,
        "use_normals": True,
        "use_uvs": True,
        "use_materials": False
    }
    bpy.ops.export_scene.obj(**export_kwargs)
    log(f"Exported OBJ: {filepath}")

# Export a vertex-group-only mesh as OBJ (duplicate head, delete other verts)
def export_vertex_group_to_obj(character_obj, vertex_group_name, filepath):
    try:
        bpy.ops.object.select_all(action='DESELECT')
        character_obj.select_set(True)
        bpy.context.view_layer.objects.active = character_obj

        # Duplicate object
        bpy.ops.object.duplicate()
        dup = bpy.context.active_object
        dup.name = f"{character_obj.name}_scalp_export"

        # Enter edit mode and delete vertices not in vertex group
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_set_active(group=vertex_group_name)
        bpy.ops.object.vertex_group_select()
        # Invert selection and delete everything else
        bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.delete(type='VERT')
        bpy.ops.object.mode_set(mode='OBJECT')

        # Export duplicate as OBJ
        export_object_as_obj(dup, filepath, apply_modifiers=False)

        # Remove duplicate from scene
        bpy.data.objects.remove(dup, do_unlink=True)
        log(f"Exported vertex-group OBJ for '{vertex_group_name}' to {filepath}")
        return True
    except Exception as e:
        log(f"Failed to export vertex group to OBJ: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return False

# Run the HairNet subprocess command
def run_hairnet_subprocess(cmd):
    try:
        log(f"Running HairNet command:\n{cmd}")
        completed = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=HAIRNET_REPO_PATH, timeout=300)
        log("HairNet stdout:\n" + completed.stdout)
        if completed.returncode != 0:
            log("HairNet stderr:\n" + completed.stderr, "ERROR")
            return False, completed.stdout + "\n" + completed.stderr
        return True, completed.stdout
    except subprocess.TimeoutExpired:
        log("HairNet process timed out", "ERROR")
        return False, "timeout"
    except Exception as e:
        log(f"Error running HairNet: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return False, str(e)

# Import the generated hair OBJ back into the Blender scene and nest it
def import_hair_obj_and_setup(obj_filepath, target_character_obj, collection_name="HairNet_Output"):
    try:
        # Import
        bpy.ops.import_scene.obj(filepath=obj_filepath)
        # Imported objects are selected; gather them
        imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
        if not imported:
            log("No mesh imported from HairNet output", "ERROR")
            return None

        # Create collection to hold imported hair
        coll = bpy.data.collections.get(collection_name)
        if not coll:
            coll = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(coll)

        for o in imported:
            # Move to collection
            for c in o.users_collection:
                c.objects.unlink(o)
            coll.objects.link(o)
            # Parent to head (keep transform)
            o.parent = target_character_obj
            o.matrix_parent_inverse = target_character_obj.matrix_world.inverted()
            o.name = f"hairnet_{o.name}"
            # Optionally add shrinkwrap to keep it on scalp
            sw = o.modifiers.new(name="HR_Shrinkwrap", type='SHRINKWRAP')
            sw.target = target_character_obj
            sw.wrap_method = 'NEAREST_SURFACEPOINT'
            sw.offset = 0.001
            log(f"Imported hair object: {o.name}")

        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        return imported
    except Exception as e:
        log(f"Failed to import hair OBJ: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return None

# High-level pipeline to run HairNet and re-import result
def run_hairnet_pipeline(character_obj, hair_obj=None, hair_params=None, prompt_text=None):
    """
    character_obj: Blender object representing head
    hair_obj: Blender object to use as initial hair (optional)
    hair_params: dict with keys like 'style' or 'length' mapped to hairnet style
    prompt_text: textual style prompt to pass to hairnet (if supported)
    """
    tmpdir = _make_temp_dir("hairnet")
    head_path = os.path.join(tmpdir, "head.obj")
    init_hair_path = os.path.join(tmpdir, "init_hair.obj")
    scalp_path = os.path.join(tmpdir, "scalp.obj")
    out_dir = os.path.join(tmpdir, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Export head
    export_object_as_obj(character_obj, head_path, apply_modifiers=True)

    # Export hair if provided, else create a simple proxy plane/object from vertex group
    if hair_obj:
        export_object_as_obj(hair_obj, init_hair_path, apply_modifiers=True)
    else:
        # create a simple placeholder export (empty OBJ) - HairNet may accept missing init mesh
        open(init_hair_path, 'w').close()
        log("No initial hair mesh provided; passing empty init mesh")

    # Ensure a vertex group for the head scalp exists (create if missing)
    vg_name = "Hair_Region"
    if vg_name not in character_obj.vertex_groups:
        create_head_vertex_group_safe(character_obj)

    if not export_vertex_group_to_obj(character_obj, "Hair_Region", scalp_path):
        log("Failed to export scalp vertex group; proceeding without scalp (may reduce quality).", "WARNING")

    # Build style string from parameters
    style = ""
    if hair_params:
        # map basic params to a simple style string. Customize as you like
        if hair_params.get("style"):
            style = hair_params.get("style")
        else:
            # Example: length and curl
            length = hair_params.get("length", hair_params.get("particle_length", 0.5))
            curl = hair_params.get("curl_intensity", 0.0)
            style = f"length={length},curl={curl}"

    if prompt_text:
        style = (style + " " + prompt_text) if style else prompt_text

    # Optional: copy reference image into tmpdir for visual guidance (if hairnet supports)
    if os.path.exists(REFERENCE_IMAGE_PATH):
        try:
            shutil.copy(REFERENCE_IMAGE_PATH, os.path.join(tmpdir, os.path.basename(REFERENCE_IMAGE_PATH)))
            log(f"Copied reference image to run dir: {REFERENCE_IMAGE_PATH}")
        except Exception as e:
            log(f"Couldn't copy reference image: {e}", "WARNING")

    # Build command (replace placeholders)
    cmd = HAIRNET_CMD_TEMPLATE.format(
        repo=HAIRNET_REPO_PATH,
        head_obj=head_path,
        hair_obj=init_hair_path,
        scalp_obj=scalp_path,
        outdir=out_dir,
        style=style.replace("'", "\"")
    )

    success, output = run_hairnet_subprocess(cmd)
    if not success:
        log("HairNet failed. See logs above.", "ERROR")
        return False

    # Attempt to find output hair OBJ (common filename - adjust if your HairNet uses different name)
    # We'll search out_dir for any OBJ and import the first one found
    hair_output_obj = None
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith(".obj"):
                hair_output_obj = os.path.join(root, f)
                break
        if hair_output_obj:
            break

    if not hair_output_obj:
        log("No hair OBJ found in HairNet output directory", "ERROR")
        return False

    imported = import_hair_obj_and_setup(hair_output_obj, character_obj, collection_name="HairNet_Output")
    if imported:
        log("✅ HairNet integration pipeline completed and hair imported.")
        return True
    else:
        log("✗ HairNet finished but import failed.", "ERROR")
        return False


def check_for_requests():
    """Timer function that checks for new character requests"""
    global is_monitoring
   
    if not is_monitoring:
        return None
    
    # Update status heartbeat
    update_status("active")
    
    try:
        if not os.path.exists(REQUEST_FILE):
            return 0.5
        
        log(f"\n{'='*70}")
        log(f"📥 NEW REQUEST DETECTED")
        log(f"{'='*70}")
        
        # Read request
        with open(REQUEST_FILE, 'r') as f:
            request_data = json.load(f)
       
        structured_data = request_data.get('structured_data', {})
        prompt = request_data.get('prompt', 'No prompt provided')
       
        if not structured_data:
            error_msg = "No structured data provided"
            log(f"✗ ERROR: {error_msg}", "ERROR")
            send_error_response(prompt, error_msg)
            return 0.5

        gender = structured_data.get('gender', 'male')
        ethnicity = structured_data.get('ethnicity', 'caucasian')
        has_hair = structured_data.get('has_hair', False)
        
        log(f"Prompt: {prompt}")
        log(f"Gender: {gender.upper()}")
        log(f"Ethnicity: {ethnicity.upper()}")
        log(f"Properties: {len(structured_data.get('properties', {}))}")
        log(f"Has Hair: {'YES' if has_hair else 'NO'}")
       
        # Get character object
        character = get_object_by_gender(gender)
       
        if not character:
            error_msg = f"Character object '{gender}' not found in scene"
            log(f"✗ ERROR: {error_msg}", "ERROR")
            send_error_response(prompt, error_msg, gender=gender)
            return 0.5
        
        # STEP 1: Apply body properties
        log(f"\nWorking on character: {character.name}")
        applied, failed = process_enhanced_properties(structured_data, character, gender)
        
        if applied == 0:
            log("⚠ WARNING: No properties were applied!", "WARNING")
        
        # STEP 2: Apply hair if requested
        hair_success = False
        hair_method = 'none'
        
                # STEP 2: Apply hair if requested
        hair_success = False
        hair_method = 'none'

        if has_hair:
            log("\nProcessing hair generation...")
            time.sleep(0.3)  # Brief pause to ensure body updates complete

            hair_params = structured_data.get('hair_params', {})
            if not hair_params:
                log("Using default hair parameters", "WARNING")
                hair_params = {
                    'particle_count': 5000,
                    'particle_length': 0.5,
                    'curl_intensity': 0.0,
                    'randomness': 0.3,
                    'engine': 'particle_system'
                }

            # Choose engine: 'hairnet' to use HairNet-AI, otherwise fallback to particle system
            requested_engine = hair_params.get('engine', 'particle_system').lower()

            if requested_engine == 'hairnet':
                # Attempt to run hairnet pipeline
                try:
                    prompt_text = request_data.get('prompt', None)
                    hair_success = run_hairnet_pipeline(character, hair_obj=None, hair_params=hair_params, prompt_text=prompt_text)
                    hair_method = 'hairnet' if hair_success else 'hairnet_failed'
                except Exception as e:
                    log(f"HairNet pipeline exception: {e}", "ERROR")
                    hair_method = 'hairnet_failed'
            else:
                hair_success = create_simple_hair_system(character, hair_params)
                hair_method = 'particle_system' if hair_success else 'failed'

            if hair_success:
                log(f"\n✅ Hair applied successfully! (method={hair_method})")
            else:
                log(f"\n⚠️  Hair generation failed (method={hair_method})", "WARNING")

        
        # Send success response
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "gender": gender,
            "ethnicity": ethnicity,
            "status": "completed",
            "message": f"✓ {gender.capitalize()} character generated!",
            "properties_applied": applied,
            "properties_failed": failed,
            "character_object": character.name,
            "has_hair": has_hair,
            "hair_method": hair_method,
            "hair_status": "Applied successfully" if hair_success else ("Failed" if has_hair else "Not requested")
        }
        
        with open(RESPONSE_FILE, 'w') as f:
            json.dump(response_data, f, indent=2)
       
        os.remove(REQUEST_FILE)
        
        log(f"\n{'='*70}")
        log(f"✅ CHARACTER GENERATION COMPLETE!")
        log(f"{'='*70}\n")
       
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        log(f"✗ {error_msg}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        send_error_response("Unknown", error_msg)
   
    return 0.5

def send_error_response(prompt, error_message, **kwargs):
    """Send error response to frontend"""
    try:
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "status": "error",
            "message": error_message,
            **kwargs
        }
        with open(RESPONSE_FILE, 'w') as f:
            json.dump(error_response, f, indent=2)
       
        if os.path.exists(REQUEST_FILE):
            os.remove(REQUEST_FILE)
    except Exception as e:
        log(f"Failed to send error response: {e}", "ERROR")

# =============================================================================
# BLENDER UI PANEL
# =============================================================================
class MESH_PT_character_bridge(bpy.types.Panel):
    bl_label = "Enhanced Character Generator Bridge"
    bl_idname = "MESH_PT_character_bridge"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="Character Generation Bridge", icon='USER')
        
        row = layout.row()
        row.scale_y = 1.5
        row.operator("mesh.start_bridge")
        
        row = layout.row()
        row.scale_y = 1.5
        row.operator("mesh.stop_bridge")
        
        # Show status
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, 'r') as f:
                    status = json.load(f)
                layout.label(text=f"Status: {status.get('status', 'unknown')}", icon='INFO')
            except:
                pass

class MESH_OT_start_bridge(bpy.types.Operator):
    bl_idname = "mesh.start_bridge"
    bl_label = "Start Bridge"
    
    def execute(self, context):
        start_bridge_monitoring()
        self.report({'INFO'}, "Bridge started")
        return {'FINISHED'}

class MESH_OT_stop_bridge(bpy.types.Operator):
    bl_idname = "mesh.stop_bridge"
    bl_label = "Stop Bridge"
    
    def execute(self, context):
        stop_bridge_monitoring()
        self.report({'INFO'}, "Bridge stopped")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(MESH_PT_character_bridge)
    bpy.utils.register_class(MESH_OT_start_bridge)
    bpy.utils.register_class(MESH_OT_stop_bridge)

def unregister():
    bpy.utils.unregister_class(MESH_PT_character_bridge)
    bpy.utils.unregister_class(MESH_OT_start_bridge)
    bpy.utils.unregister_class(MESH_OT_stop_bridge)

# Initialize
register()

log("="*70)
log("🎭 BLENDER CHARACTER GENERATOR BRIDGE")
log("="*70)
log("Ready! Click 'Start Bridge' in the Properties panel.")
log("="*70)