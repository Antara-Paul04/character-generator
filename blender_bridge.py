import bpy
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

def check_for_requests():
    """Timer function that checks for new character requests."""
    global is_monitoring
   
    if not is_monitoring:
        return None
    
    try:
        if os.path.exists(REQUEST_FILE):
            # Read the request
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

            # Extract gender and ethnicity from structured data
            gender = structured_data.get('gender', 'male')
            ethnicity = structured_data.get('ethnicity', 'caucasian')
            
            print(f"\n{'='*60}")
            print(f"🎭 NEW CHARACTER REQUEST")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"Gender: {gender.upper()}")
            print(f"Ethnicity: {ethnicity.upper()}")
            print(f"Properties to apply: {len(structured_data.get('properties', {}))}")
            print(f"{'='*60}\n")
           
            # Get the appropriate character object based on gender
            character = get_object_by_gender(gender)
           
            if character:
                applied, failed = process_enhanced_properties(structured_data, character, gender)
               
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "status": "completed",
                    "message": f"✓ {gender.capitalize()} character generated successfully!",
                    "properties_applied": applied,
                    "properties_failed": failed,
                    "character_object": character.name
                }
            else:
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "status": "error",
                    "message": f"Could not find {gender} character object in Blender scene."
                }
           
            # Send response
            with open(RESPONSE_FILE, 'w') as f:
                json.dump(response_data, f)
           
            # Remove request file
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