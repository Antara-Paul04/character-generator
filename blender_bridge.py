import bpy
from mathutils import Vector
import re
import json
import os
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import traceback

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
def create_simple_hair_system(character_obj, hair_params):
    """Create particle hair system - SIMPLIFIED AND ROBUST"""
    try:
        log(f"\n{'='*60}")
        log(f"STEP 2: CREATING HAIR SYSTEM")
        log(f"{'='*60}")
        
        # Make character active
        bpy.context.view_layer.objects.active = character_obj
        character_obj.select_set(True)
        log(f"Set active object: {character_obj.name}")
        
        # Remove any existing hair systems
        while len(character_obj.particle_systems) > 0:
            bpy.ops.object.particle_system_remove()
        log("Cleared existing particle systems")
        
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
        
        # Add children for volume
        settings.child_nbr = max(100, int(settings.count * 0.1))
        settings.child_length = 1.0
        settings.child_radius = float(randomness)
        
        log(f"✓ Added child particles: {settings.child_nbr}")
        
        # Set curl if needed
        if curl_intensity > 0:
            settings.child_type = 'INTERPOLATED'
            settings.clump_factor = 0.2 + curl_intensity * 0.3
            settings.roughness_1 = curl_intensity
            log(f"✓ Set curl intensity: {curl_intensity}")
        
        # Try to limit to head region (optional, won't fail if doesn't work)
        try:
            vertex_group = create_head_vertex_group_safe(character_obj)
            if vertex_group:
                psys.vertex_group_density = vertex_group
                log(f"✓ Limited hair to vertex group: {vertex_group}")
        except Exception as e:
            log(f"⚠ Could not limit hair to head (will cover whole body): {e}", "WARNING")
        
        # Force update
        bpy.context.view_layer.update()
        
        log(f"\n✅ HAIR SYSTEM CREATED SUCCESSFULLY!")
        log(f"   System name: {psys.name}")
        log(f"   Particle count: {settings.count}")
        log(f"   Hair length: {settings.hair_length}")
        
        return True
        
    except Exception as e:
        log(f"✗ Hair creation failed: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return False

def create_head_vertex_group_safe(character_obj):
    """Create vertex group for head region - safe version that won't crash"""
    try:
        mesh = character_obj.data
        
        # Remove existing hair vertex group
        if "Hair_Region" in character_obj.vertex_groups:
            character_obj.vertex_groups.remove(character_obj.vertex_groups["Hair_Region"])
        
        hair_group = character_obj.vertex_groups.new(name="Hair_Region")
        
        # Simple approach: select top 15% of vertices by Z coordinate
        vertices = mesh.vertices
        world_matrix = character_obj.matrix_world
        
        # Get all Z coordinates
        z_coords = [(i, (world_matrix @ v.co).z) for i, v in enumerate(vertices)]
        z_coords.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 15%
        num_head_verts = max(100, int(len(vertices) * 0.15))
        head_vertex_indices = [idx for idx, z in z_coords[:num_head_verts]]
        
        # Add to group
        hair_group.add(head_vertex_indices, 1.0, 'ADD')
        
        log(f"✓ Created Hair_Region with {len(head_vertex_indices)} vertices")
        return "Hair_Region"
        
    except Exception as e:
        log(f"⚠ Vertex group creation failed: {e}", "WARNING")
        return None

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
        
        if has_hair:
            log("\nProcessing hair generation...")
            time.sleep(0.3)  # Brief pause to ensure body updates complete
            
            # Get hair parameters
            hair_params = structured_data.get('hair_params', {})
            if not hair_params:
                log("Using default hair parameters", "WARNING")
                hair_params = {
                    'particle_count': 5000,
                    'particle_length': 0.5,
                    'curl_intensity': 0.0,
                    'randomness': 0.3
                }
            
            hair_success = create_simple_hair_system(character, hair_params)
            hair_method = 'particle_system' if hair_success else 'failed'
            
            if hair_success:
                log(f"\n✅ Hair applied successfully!")
            else:
                log(f"\n⚠️  Hair generation failed", "WARNING")
        else:
            log("\nℹ️  No hair requested")
        
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