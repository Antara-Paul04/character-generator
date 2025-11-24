import bpy
from mathutils import Vector  # FIXED: was mathlib
import re
import json
import os
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional

# =============================================================================
# BLENDER BRIDGE CONFIGURATION
# =============================================================================
COMMUNICATION_DIR = r"C:\temp\blender_bridge"
REQUEST_FILE = os.path.join(COMMUNICATION_DIR, "character_request.json")
RESPONSE_FILE = os.path.join(COMMUNICATION_DIR, "character_response.json")

is_monitoring = False

# =============================================================================
# BLENDER HELPER FUNCTIONS
# =============================================================================
def get_object_by_gender(gender="male"):
    """Get the appropriate character object based on detected gender."""
    base_name = "mb_female" if gender == "female" else "mb_male"
    
    if base_name in bpy.data.objects:
        return bpy.data.objects[base_name]
    
    for obj in bpy.data.objects:
        if obj.name.startswith(base_name):
            print(f"✓ Found character object: {obj.name}")
            return obj
    
    fallback_name = "mb_male" if gender == "female" else "mb_female"
    print(f"⚠️  {base_name} not found, trying fallback: {fallback_name}")
    
    if fallback_name in bpy.data.objects:
        return bpy.data.objects[fallback_name]
    
    for obj in bpy.data.objects:
        if obj.name.startswith(fallback_name):
            print(f"⚠️  Using fallback object: {obj.name}")
            return obj
    
    print(f"✗ Could not find any character object (tried {base_name} and {fallback_name})")
    return None

def reset_character_shape_keys(obj):
    """Resets all shape key values to 0.0 for a clean start."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        return
    
    print("--- Resetting all shape keys ---")
    reset_count = 0
    for kb in obj.data.shape_keys.key_blocks:
        if kb.name != "Basis":
            kb.value = 0.0
            reset_count += 1
    print(f"✓ Reset {reset_count} shape keys")

def apply_morph(obj, shape_key_name, value):
    """Applies a single morph value, checking if the key exists."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        return False
    
    if shape_key_name in obj.data.shape_keys.key_blocks:
        obj.data.shape_keys.key_blocks[shape_key_name].value = min(1.0, max(0.0, value))
        return True
    else:
        return False

# =============================================================================
# BODY PROPERTY PROCESSING (STEP 1)
# =============================================================================
def process_enhanced_properties(structured_data: Dict, character_obj, gender: str):
    """
    STEP 1: Applies body morphs from enhanced property mapping
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: APPLYING BODY PROPERTIES")
    print(f"{'='*60}")
    print(f"Processing {gender.upper()} character...")
    
    reset_character_shape_keys(character_obj)
    
    properties = structured_data.get("properties", {})
    analysis = structured_data.get("analysis", {})
    
    print(f"Total properties to apply: {len(properties)}")
    
    applied_count = 0
    failed_count = 0
    
    for property_name, intensity in properties.items():
        if apply_morph(character_obj, property_name, intensity):
            applied_count += 1
        else:
            failed_count += 1
    
    # CRITICAL: Update the view layer to apply changes
    bpy.context.view_layer.update()
    
    print(f"✓ Body properties applied: {applied_count}")
    print(f"✗ Body properties failed: {failed_count}")
    
    if analysis:
        print(f"\n📊 Analysis: {analysis.get('analysis', 'N/A')}")
    
    return applied_count, failed_count

# =============================================================================
# HAIR GENERATION (STEP 2)
# =============================================================================
def create_head_vertex_group(character_obj):
    """Create a vertex group for SCALP ONLY"""
    try:
        mesh = character_obj.data
        
        # Remove existing hair vertex group
        if "Hair_Region" in character_obj.vertex_groups:
            character_obj.vertex_groups.remove(character_obj.vertex_groups["Hair_Region"])
        
        hair_group = character_obj.vertex_groups.new(name="Hair_Region")
        vertices = mesh.vertices
        world_matrix = character_obj.matrix_world
        
        # Calculate head region
        z_coords = [world_matrix @ v.co for v in vertices]
        max_z = max(z.z for z in z_coords)
        min_z = min(z.z for z in z_coords)
        height_range = max_z - min_z
        
        # Top 20% of character = head/scalp area
        head_threshold = max_z - (height_range * 0.20)
        
        print(f"  Character height: {height_range:.3f}m")
        print(f"  Head threshold Z: {head_threshold:.3f}m")
        
        head_vertex_indices = []
        
        for i, vertex in enumerate(vertices):
            world_pos = world_matrix @ vertex.co
            
            # Check if vertex is in head region
            if world_pos.z > head_threshold:
                # Only include vertices near center (scalp, not ears/face)
                x_dist = abs(world_pos.x)
                y_dist = abs(world_pos.y)
                
                # Tight bounds for scalp only
                if x_dist < 0.12 and y_dist < 0.12:
                    head_vertex_indices.append(i)
        
        if len(head_vertex_indices) < 100:
            print("⚠️  Too few vertices, relaxing constraints...")
            head_vertex_indices = []
            for i, vertex in enumerate(vertices):
                world_pos = world_matrix @ vertex.co
                if world_pos.z > head_threshold:
                    head_vertex_indices.append(i)
        
        hair_group.add(head_vertex_indices, 1.0, 'ADD')
        
        print(f"✓ Created Hair_Region with {len(head_vertex_indices)} vertices")
        return "Hair_Region"
        
    except Exception as e:
        print(f"✗ Error creating vertex group: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_particle_hair_system(character_obj, hair_params):
    """Create particle hair system on HEAD ONLY"""
    try:
        print(f"\n{'='*60}")
        print(f"STEP 2: CREATING HAIR SYSTEM")
        print(f"{'='*60}")
        
        bpy.context.view_layer.objects.active = character_obj
        character_obj.select_set(True)
        
        # Step 1: Create vertex group for head
        print("Creating head vertex group...")
        vertex_group_name = create_head_vertex_group(character_obj)
        
        if not vertex_group_name:
            print("⚠️  Vertex group creation failed, hair will cover whole body")
        
        # Step 2: Add particle system
        print("Adding particle system...")
        bpy.ops.object.particle_system_add()
        
        psys = character_obj.particle_systems[-1]
        psys.name = "Hair_System"
        
        settings = psys.settings
        settings.type = 'HAIR'
        
        # Step 3: CRITICAL - Assign vertex group to limit hair to head
        if vertex_group_name:
            psys.vertex_group_density = vertex_group_name
            print(f"✓ Hair limited to: {vertex_group_name}")
        
        # Step 4: Configure hair parameters
        settings.count = hair_params.get('particle_count', 5000)
        settings.hair_length = hair_params.get('particle_length', 0.5)
        settings.hair_step = 5
        settings.render_step = 5
        
        # Children for volume
        settings.child_nbr = int(settings.count * 0.1)
        settings.child_length = 1.0
        settings.child_radius = hair_params.get('randomness', 0.3)
        
        # Curl settings
        curl_intensity = hair_params.get('curl_intensity', 0.0)
        if curl_intensity > 0:
            settings.child_type = 'INTERPOLATED'
            settings.clump_factor = 0.2 + curl_intensity * 0.3
            settings.roughness_1 = curl_intensity
        
        print(f"✓ Hair created with {settings.count} particles")
        print(f"  Length: {settings.hair_length}")
        print(f"  Curl: {curl_intensity}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hair creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def apply_hair_to_character(character_obj, structured_data):
    """Apply hair to character - called after body properties"""
    result = {
        'success': False,
        'method': None,
        'message': ''
    }
    
    # Extract hair parameters from structured_data
    hair_params = structured_data.get('hair_params', {})
    hair_generation = structured_data.get('hair_generation', {})
    
    print(f"\n💇 Hair application starting...")
    print(f"  Has hair_params: {bool(hair_params)}")
    print(f"  Hair generation: {hair_generation.get('success', False)}")
    
    if not hair_params:
        result['message'] = 'No hair parameters provided'
        return result
    
    # Method 1: Try HairNet OBJ import (if available)
    if hair_generation.get('success') and not hair_generation.get('mock'):
        obj_path = hair_generation.get('output_path')
        if obj_path and os.path.exists(obj_path):
            print(f"Attempting to import HairNet OBJ: {obj_path}")
            # Import OBJ code here if needed
            pass
    
    # Method 2: Use particle system (primary method)
    print("Using particle system for hair generation...")
    
    # Extract particle parameters
    particle_params = {
        'particle_count': hair_params.get('particle_count', 5000),
        'particle_length': hair_params.get('particle_length', 0.5),
        'curl_intensity': hair_params.get('curl_intensity', 0.0),
        'randomness': hair_params.get('randomness', 0.3)
    }
    
    print(f"Particle params: {particle_params}")
    
    if create_particle_hair_system(character_obj, particle_params):
        result['success'] = True
        result['method'] = 'particle_system'
        result['message'] = f"Hair created with {particle_params['particle_count']} particles"
        return result
    
    result['message'] = 'Hair creation failed'
    return result

# =============================================================================
# BRIDGE MONITORING
# =============================================================================
def start_bridge_monitoring():
    """Start monitoring for character generation requests."""
    global is_monitoring
   
    if is_monitoring:
        print("Bridge monitoring is already active.")
        return
   
    os.makedirs(COMMUNICATION_DIR, exist_ok=True)
   
    print(f"🎭 Starting Enhanced Blender Bridge...")
    print(f"📁 Watching: {COMMUNICATION_DIR}")
    print("⏳ Ready for requests...")
   
    is_monitoring = True
    bpy.app.timers.register(check_for_requests, first_interval=0.5)

def stop_bridge_monitoring():
    """Stop monitoring for requests."""
    global is_monitoring
    is_monitoring = False
   
    if bpy.app.timers.is_registered(check_for_requests):
        bpy.app.timers.unregister(check_for_requests)
   
    print("Bridge monitoring stopped.")

def check_for_requests():
    """Timer function that checks for new character requests"""
    global is_monitoring
   
    if not is_monitoring:
        return None
    
    try:
        if os.path.exists(REQUEST_FILE):
            with open(REQUEST_FILE, 'r') as f:
                request_data = json.load(f)
           
            structured_data = request_data.get('structured_data', {})
            prompt = request_data['prompt']
           
            if not structured_data:
                print("✗ Error: Missing structured_data")
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "status": "error",
                    "message": "No structured data provided"
                }
                with open(RESPONSE_FILE, 'w') as f:
                    json.dump(response_data, f)
                os.remove(REQUEST_FILE)
                return 0.5

            gender = structured_data.get('gender', 'male')
            ethnicity = structured_data.get('ethnicity', 'caucasian')
            has_hair = structured_data.get('has_hair', False)
            
            print(f"\n{'='*70}")
            print(f"🎭 NEW CHARACTER REQUEST")
            print(f"{'='*70}")
            print(f"Prompt: {prompt}")
            print(f"Gender: {gender.upper()}")
            print(f"Ethnicity: {ethnicity.upper()}")
            print(f"Properties: {len(structured_data.get('properties', {}))}")
            print(f"Has Hair: {'YES' if has_hair else 'NO'}")
            print(f"{'='*70}")
           
            character = get_object_by_gender(gender)
           
            if not character:
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "status": "error",
                    "message": f"Character object '{gender}' not found in scene"
                }
                with open(RESPONSE_FILE, 'w') as f:
                    json.dump(response_data, f)
                os.remove(REQUEST_FILE)
                return 0.5
            
            # STEP 1: Apply body properties FIRST
            print(f"\nWorking on character: {character.name}")
            applied, failed = process_enhanced_properties(structured_data, character, gender)
            
            # STEP 2: Apply hair AFTER body is complete
            hair_result = None
            if has_hair:
                time.sleep(0.5)  # Brief pause to ensure body properties are fully applied
                hair_result = apply_hair_to_character(character, structured_data)
                
                if hair_result['success']:
                    print(f"\n✅ Hair applied: {hair_result['method']}")
                else:
                    print(f"\n⚠️  Hair failed: {hair_result['message']}")
            else:
                print("\nℹ️  No hair requested")
            
            # Send response
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
                "hair_method": hair_result['method'] if hair_result and hair_result['success'] else 'none',
                "hair_status": hair_result['message'] if hair_result else 'No hair requested'
            }
            
            with open(RESPONSE_FILE, 'w') as f:
                json.dump(response_data, f)
           
            os.remove(REQUEST_FILE)
            
            print(f"\n{'='*70}")
            print(f"✅ CHARACTER GENERATION COMPLETE!")
            print(f"{'='*70}\n")
           
    except Exception as e:
        print(f"✗ Error processing request: {e}")
        import traceback
        traceback.print_exc()
        
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "message": f"Error: {str(e)}"
        }
        with open(RESPONSE_FILE, 'w') as f:
            json.dump(error_response, f)
       
        if os.path.exists(REQUEST_FILE):
            os.remove(REQUEST_FILE)
   
    return 0.5

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

register()

print("="*70)
print("🎭 BLENDER CHARACTER GENERATOR BRIDGE")
print("="*70)
print("Ready! Click 'Start Bridge' to begin.")
print("="*70)