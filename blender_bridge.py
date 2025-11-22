import bpy
from mathlib import Vector
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
    
    # Try to find the exact object
    if base_name in bpy.data.objects:
        return bpy.data.objects[base_name]
    
    # Try to find object with similar name
    for obj in bpy.data.objects:
        if obj.name.startswith(base_name):
            print(f"✓ Found character object: {obj.name}")
            return obj
    
    # If gender-specific not found, try the opposite gender
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
        if kb.name != "Basis":  # Don't reset the Basis shape key
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
# ENHANCED PROPERTY PROCESSING
# =============================================================================
def process_enhanced_properties(structured_data: Dict, character_obj, gender: str):
    """
    Applies morphs from enhanced property mapping with gender awareness.
    """
    print(f"--- Starting Enhanced Property Processing for {gender.upper()} character ---")
    reset_character_shape_keys(character_obj)
    
    properties = structured_data.get("properties", {})
    analysis = structured_data.get("analysis", {})
    
    print(f"Processing {len(properties)} properties from analysis...")
    
    # Apply each property with its intensity value
    applied_count = 0
    failed_count = 0
    applied_properties = []
    failed_properties = []
    
    for property_name, intensity in properties.items():
        if apply_morph(character_obj, property_name, intensity):
            applied_count += 1
            applied_properties.append(f"{property_name} ({intensity:.2f})")
        else:
            failed_count += 1
            failed_properties.append(property_name)
    
    bpy.context.view_layer.update()
    
    print(f"\n✓ Enhanced character generation complete!")
    print(f"  - Applied {applied_count} properties successfully")
    if failed_count > 0:
        print(f"  - Failed to apply {failed_count} properties (not found in model)")
        if failed_count <= 10:  # Only show failed properties if not too many
            print(f"  - Failed properties: {', '.join(failed_properties[:10])}")
    
    # Print analysis summary
    if analysis:
        print("\n📊 Analysis Summary:")
        if analysis.get('analysis'):
            print(f"  - {analysis['analysis']}")
        if analysis.get('cultural_context'):
            print(f"  - {analysis['cultural_context']}")
        if analysis.get('lifestyle_traits'):
            print(f"  - {analysis['lifestyle_traits']}")
    
    return applied_count, failed_count

# =============================================================================
# BRIDGE MONITORING FUNCTIONS
# =============================================================================
def start_bridge_monitoring():
    """Start monitoring for character generation requests."""
    global is_monitoring
   
    if is_monitoring:
        print("Bridge monitoring is already active.")
        return
   
    os.makedirs(COMMUNICATION_DIR, exist_ok=True)
   
    print(f"🎭 Starting Enhanced Blender Bridge with Gender Detection...")
    print(f"📁 Watching directory: {COMMUNICATION_DIR}")
    print("⏳ Waiting for character generation requests...")
   
    is_monitoring = True
    bpy.app.timers.register(check_for_requests, first_interval=0.5)

def stop_bridge_monitoring():
    """Stop monitoring for requests."""
    global is_monitoring
    is_monitoring = False
   
    if bpy.app.timers.is_registered(check_for_requests):
        bpy.app.timers.unregister(check_for_requests)
   
    print("Bridge monitoring stopped.")

def import_hair_obj(obj_path, character_obj):
    """Import hair .obj file and attach it to character"""
    try:
        if not os.path.exists(obj_path):
            print(f"✗ Hair OBJ not found: {obj_path}")
            return False
        
        # Import the OBJ
        bpy.ops.import_scene.obj(filepath=obj_path)
        
        # Get the imported objects
        imported_objects = [obj for obj in bpy.context.selected_objects]
        
        if not imported_objects:
            print("✗ No objects were imported from hair OBJ")
            return False
        
        print(f"✓ Imported {len(imported_objects)} hair objects")
        
        # Find the character's head location
        head_location = get_head_location(character_obj)
        
        # Parent all hair objects to the character
        for hair_obj in imported_objects:
            hair_obj.location = head_location
            hair_obj.parent = character_obj
            hair_obj.matrix_parent_inverse = character_obj.matrix_world.inverted()
            hair_obj.name = f"{character_obj.name}_hair_{hair_obj.name}"
            
            print(f"✓ Attached {hair_obj.name} to {character_obj.name}")
        
        setup_hair_material(imported_objects)
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing hair: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_head_location(character_obj):
    """Find the approximate head location of the character"""
    if not character_obj or not character_obj.data:
        return Vector((0, 0, 1.7))
    
    mesh = character_obj.data
    world_matrix = character_obj.matrix_world
    
    max_z = -float('inf')
    head_location = Vector((0, 0, 0))
    
    for vertex in mesh.vertices:
        world_coord = world_matrix @ vertex.co
        if world_coord.z > max_z:
            max_z = world_coord.z
            head_location = world_coord
    
    head_location.z += 0.1
    
    print(f"✓ Head location: {head_location}")
    return head_location

def setup_hair_material(hair_objects):
    """Setup basic material for hair objects"""
    try:
        mat_name = "Hair_Material"
        
        if mat_name not in bpy.data.materials:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Base Color'].default_value = (0.05, 0.02, 0.01, 1)
                bsdf.inputs['Roughness'].default_value = 0.4
                bsdf.inputs['Specular'].default_value = 0.5
        else:
            mat = bpy.data.materials[mat_name]
        
        for obj in hair_objects:
            if obj.type == 'MESH':
                if len(obj.data.materials) == 0:
                    obj.data.materials.append(mat)
                else:
                    obj.data.materials[0] = mat
        
        print(f"✓ Applied hair material to {len(hair_objects)} objects")
        
    except Exception as e:
        print(f"⚠ Warning: Could not setup hair material: {e}")

def create_particle_hair_system(character_obj, hair_params):
    """Create a Blender particle hair system"""
    try:
        bpy.context.view_layer.objects.active = character_obj
        character_obj.select_set(True)
        
        bpy.ops.object.particle_system_add()
        
        psys = character_obj.particle_systems[-1]
        psys.name = "Hair_System"
        
        settings = psys.settings
        settings.type = 'HAIR'
        
        settings.count = hair_params.get('particle_count', 5000)
        settings.hair_length = hair_params.get('particle_length', 0.5)
        settings.hair_step = 5
        settings.render_step = 5
        
        settings.child_nbr = int(hair_params.get('particle_count', 5000) * 0.1)
        settings.child_length = 1.0
        settings.child_radius = hair_params.get('randomness', 0.3)
        
        curl_intensity = hair_params.get('curl_intensity', 0.0)
        if curl_intensity > 0:
            settings.child_type = 'INTERPOLATED'
            settings.clump_factor = 0.2 + curl_intensity * 0.3
            settings.roughness_1 = curl_intensity
        
        print(f"✓ Created particle hair system with {settings.count} particles")
        return True
        
    except Exception as e:
        print(f"✗ Error creating particle hair system: {e}")
        return False

def apply_hair_to_character(character_obj, hair_data):
    """Apply hair to character using available methods"""
    result = {
        'success': False,
        'method': None,
        'message': ''
    }
    
    # Method 1: Try to import HairNet .obj file
    if hair_data.get('hair_generation') and hair_data['hair_generation'].get('success'):
        gen_result = hair_data['hair_generation']
        if 'output_path' in gen_result and not gen_result.get('mock', False):
            obj_path = gen_result['output_path']
            if import_hair_obj(obj_path, character_obj):
                result['success'] = True
                result['method'] = 'hairnet_obj'
                result['message'] = 'HairNet OBJ imported successfully'
                return result
    
    # Method 2: Fallback to particle system
    if hair_data.get('hair_params'):
        if create_particle_hair_system(character_obj, hair_data['hair_params']):
            result['success'] = True
            result['method'] = 'particle_system'
            result['message'] = 'Particle hair system created'
            return result
    
    # Method 3: Create basic default hair
    default_params = {
        'particle_count': 5000,
        'particle_length': 0.5,
        'curl_intensity': 0.0,
        'randomness': 0.3
    }
    if create_particle_hair_system(character_obj, default_params):
        result['success'] = True
        result['method'] = 'default_particle'
        result['message'] = 'Default particle hair created'
        return result
    
    result['message'] = 'All hair application methods failed'
    return result

# Update the check_for_requests function to handle hair data
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
                print("✗ Error: Received request file missing 'structured_data'.")
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "status": "error",
                    "message": "Frontend failed to provide structured data."
                }
                with open(RESPONSE_FILE, 'w') as f:
                    json.dump(response_data, f)
                os.remove(REQUEST_FILE)
                return 0.5

            gender = structured_data.get('gender', 'male')
            ethnicity = structured_data.get('ethnicity', 'caucasian')
            has_hair = structured_data.get('has_hair', False)
            
            print(f"\n{'='*60}")
            print(f"🎭 NEW CHARACTER REQUEST")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"Gender: {gender.upper()}")
            print(f"Ethnicity: {ethnicity.upper()}")
            print(f"Properties: {len(structured_data.get('properties', {}))}")
            print(f"Has Hair: {'YES' if has_hair else 'NO'}")
            print(f"{'='*60}\n")
           
            character = get_object_by_gender(gender)
           
            if character:
                # Apply body properties
                applied, failed = process_enhanced_properties(structured_data, character, gender)
                
                # Apply hair if available
                hair_result = None
                if has_hair:
                    print("\n💇 Applying hair to character...")
                    hair_result = apply_hair_to_character(character, structured_data)
                    
                    if hair_result['success']:
                        print(f"✓ Hair applied using method: {hair_result['method']}")
                    else:
                        print(f"⚠ Hair application failed: {hair_result['message']}")
               
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "status": "completed",
                    "message": f"✓ {gender.capitalize()} character generated successfully!",
                    "properties_applied": applied,
                    "properties_failed": failed,
                    "character_object": character.name,
                    "has_hair": has_hair,
                    "hair_method": hair_result['method'] if hair_result and hair_result['success'] else 'none',
                    "hair_status": hair_result['message'] if hair_result else 'No hair data provided'
                }
            else:
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "status": "error",
                    "message": f"Could not find {gender} character object in Blender scene."
                }
           
            with open(RESPONSE_FILE, 'w') as f:
                json.dump(response_data, f)
           
            os.remove(REQUEST_FILE)
           
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

# def check_for_requests():
#     """Timer function that checks for new character requests."""
#     global is_monitoring
   
#     if not is_monitoring:
#         return None
    
#     try:
#         if os.path.exists(REQUEST_FILE):
#             # Read the request
#             with open(REQUEST_FILE, 'r') as f:
#                 request_data = json.load(f)
           
#             structured_data = request_data.get('structured_data', {})
#             prompt = request_data['prompt']
           
#             if not structured_data:
#                 print("✗ Error: Received request file missing 'structured_data'.")
#                 response_data = {
#                     "timestamp": datetime.now().isoformat(),
#                     "prompt": prompt,
#                     "status": "error",
#                     "message": "Frontend failed to provide structured data."
#                 }
#                 with open(RESPONSE_FILE, 'w') as f:
#                     json.dump(response_data, f)
#                 os.remove(REQUEST_FILE)
#                 return 0.5

#             # Extract gender and ethnicity from structured data
#             gender = structured_data.get('gender', 'male')
#             ethnicity = structured_data.get('ethnicity', 'caucasian')
            
#             print(f"\n{'='*60}")
#             print(f"🎭 NEW CHARACTER REQUEST")
#             print(f"{'='*60}")
#             print(f"Prompt: {prompt}")
#             print(f"Gender: {gender.upper()}")
#             print(f"Ethnicity: {ethnicity.upper()}")
#             print(f"Properties to apply: {len(structured_data.get('properties', {}))}")
#             print(f"{'='*60}\n")
           
#             # Get the appropriate character object based on gender
#             character = get_object_by_gender(gender)
           
#             if character:
#                 applied, failed = process_enhanced_properties(structured_data, character, gender)
               
#                 response_data = {
#                     "timestamp": datetime.now().isoformat(),
#                     "prompt": prompt,
#                     "gender": gender,
#                     "ethnicity": ethnicity,
#                     "status": "completed",
#                     "message": f"✓ {gender.capitalize()} character generated successfully!",
#                     "properties_applied": applied,
#                     "properties_failed": failed,
#                     "character_object": character.name
#                 }
#             else:
#                 response_data = {
#                     "timestamp": datetime.now().isoformat(),
#                     "prompt": prompt,
#                     "gender": gender,
#                     "status": "error",
#                     "message": f"Could not find {gender} character object in Blender scene."
#                 }
           
#             # Send response
#             with open(RESPONSE_FILE, 'w') as f:
#                 json.dump(response_data, f)
           
#             # Remove request file
#             os.remove(REQUEST_FILE)
           
#     except Exception as e:
#         print(f"✗ Error processing request: {e}")
#         import traceback
#         traceback.print_exc()
        
#         error_response = {
#             "timestamp": datetime.now().isoformat(),
#             "status": "error",
#             "message": f"Error: {str(e)}"
#         }
#         with open(RESPONSE_FILE, 'w') as f:
#             json.dump(error_response, f)
       
#         if os.path.exists(REQUEST_FILE):
#             os.remove(REQUEST_FILE)
   
#     return 0.5

# =============================================================================
# BLENDER UI PANEL
# =============================================================================
class MESH_PT_character_bridge(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
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
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Testing", icon='EXPERIMENTAL')
        row = box.row()
        row.operator("mesh.test_enhanced_generation")
        row = box.row()
        row.operator("mesh.test_female_generation")

class MESH_OT_start_bridge(bpy.types.Operator):
    """Start the bridge monitoring."""
    bl_idname = "mesh.start_bridge"
    bl_label = "Start Bridge"
    
    def execute(self, context):
        start_bridge_monitoring()
        self.report({'INFO'}, "Enhanced bridge monitoring started")
        return {'FINISHED'}

class MESH_OT_stop_bridge(bpy.types.Operator):
    """Stop the bridge monitoring."""
    bl_idname = "mesh.stop_bridge"
    bl_label = "Stop Bridge"
    
    def execute(self, context):
        stop_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring stopped")
        return {'FINISHED'}

class MESH_OT_test_enhanced_generation(bpy.types.Operator):
    """Test enhanced character generation with male sample."""
    bl_idname = "mesh.test_enhanced_generation"
    bl_label = "Test Male Generation"
    
    def execute(self, context):
        test_structured_data = {
            "properties": {
                "L1_Asian": 0.8,
                "L2__Eyes_Size_max": 0.7,
                "L2_Asian_Nose_TipSize_min": 0.6,
                "L2_Asian_Jaw_Angle_max": 0.8,
                "L2__Body_Size_max": 0.6,
                "L2__Arms_UpperarmMass-UpperarmTone_max-max": 0.7
            },
            "analysis": {
                "analysis": "Test: Athletic Asian male"
            },
            "gender": "male",
            "ethnicity": "asian"
        }
        
        character = get_object_by_gender("male")
        
        if character:
            process_enhanced_properties(test_structured_data, character, "male")
            self.report({'INFO'}, "Male test generation completed!")
        else:
            self.report({'ERROR'}, "Male character object not found")
        return {'FINISHED'}

class MESH_OT_test_female_generation(bpy.types.Operator):
    """Test enhanced character generation with female sample."""
    bl_idname = "mesh.test_female_generation"
    bl_label = "Test Female Generation"
    
    def execute(self, context):
        test_structured_data = {
            "properties": {
                "L1_Caucasian": 0.8,
                "L2__Eyes_Size_max": 0.8,
                "L2_Caucasian_Mouth_UpperlipVolume_max": 0.7,
                "L2__Body_Size_min": 0.5,
                "L2_Caucasian_Cheeks_Zygom_max": 0.7
            },
            "analysis": {
                "analysis": "Test: Slender Caucasian female"
            },
            "gender": "female",
            "ethnicity": "caucasian"
        }
        
        character = get_object_by_gender("female")
        
        if character:
            process_enhanced_properties(test_structured_data, character, "female")
            self.report({'INFO'}, "Female test generation completed!")
        else:
            self.report({'ERROR'}, "Female character object not found")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(MESH_PT_character_bridge)
    bpy.utils.register_class(MESH_OT_start_bridge)
    bpy.utils.register_class(MESH_OT_stop_bridge)
    bpy.utils.register_class(MESH_OT_test_enhanced_generation)
    bpy.utils.register_class(MESH_OT_test_female_generation)

def unregister():
    bpy.utils.unregister_class(MESH_PT_character_bridge)
    bpy.utils.unregister_class(MESH_OT_start_bridge)
    bpy.utils.unregister_class(MESH_OT_stop_bridge)
    bpy.utils.unregister_class(MESH_OT_test_enhanced_generation)
    bpy.utils.unregister_class(MESH_OT_test_female_generation)

# Register classes
register()

# =============================================================================
# AUTO-START AND MAIN EXECUTION
# =============================================================================
print("="*70)
print("🎭 ENHANCED CHARACTER GENERATOR BRIDGE WITH GENDER DETECTION")
print("="*70)
print("Features:")
print("  ✓ Automatic gender detection (male/female)")
print("  ✓ Intelligent ethnicity mapping with best-fit fallback")
print("  ✓ Minimum 30 properties per character")
print("  ✓ NLP + LLM enhanced analysis")
print("  ✓ Cultural context awareness")
print()
print("Available Character Objects:")
for obj in bpy.data.objects:
    if 'mb_' in obj.name.lower():
        print(f"  ✓ {obj.name}")
print()
print("Instructions:")
print("1. Run the enhanced frontend.py script")
print("2. Click 'Start Bridge' in Blender")
print("3. Open http://127.0.0.1:5000")
print("4. Generate male or female characters with detailed descriptions!")
print("="*70)

# Optional: Uncomment to auto-start monitoring
# start_bridge_monitoring()