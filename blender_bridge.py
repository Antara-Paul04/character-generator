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
def get_object(name="mb_male"):
    """Safely gets the character object from the scene."""
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            return obj
    return None

def reset_character_shape_keys(obj):
    """Resets all shape key values to 0.0 for a clean start."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        return
    print("--- Resetting all shape keys ---")
    for kb in obj.data.shape_keys.key_blocks:
        kb.value = 0.0
    print("Reset complete.")

def apply_morph(obj, shape_key_name, value):
    """Applies a single morph value, checking if the key exists."""
    if not obj or not getattr(obj.data, "shape_keys", None):
        return
    if shape_key_name in obj.data.shape_keys.key_blocks:
        print(f"Applying morph: '{shape_key_name}' with value {value:.2f}")
        obj.data.shape_keys.key_blocks[shape_key_name].value = min(1.0, value)
    else:
        print(f"Warning: Shape key '{shape_key_name}' not found.")

# =============================================================================
# ENHANCED PROPERTY PROCESSING
# =============================================================================
def process_enhanced_properties(structured_data: Dict, character_obj):
    """
    Applies morphs from enhanced property mapping (NLP + LLM analysis).
    This is the NEW function that processes the property-value mapping.
    """
    print("--- Starting Enhanced Property Processing ---")
    reset_character_shape_keys(character_obj)
    
    properties = structured_data.get("properties", {})
    analysis = structured_data.get("analysis", {})
    
    print(f"Processing {len(properties)} properties from analysis...")
    
    # Apply each property with its intensity value
    applied_count = 0
    for property_name, intensity in properties.items():
        apply_morph(character_obj, property_name, intensity)
        applied_count += 1
    
    bpy.context.view_layer.update()
    print(f"--- Enhanced character generation complete! Applied {applied_count} properties ---")
    
    # Print analysis summary
    if analysis:
        print("Analysis Summary:")
        print(f"  - {analysis.get('analysis', 'No analysis available')}")
        if analysis.get('cultural_context'):
            print(f"  - {analysis['cultural_context']}")
        if analysis.get('lifestyle_traits'):
            print(f"  - {analysis['lifestyle_traits']}")

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
   
    print(f"Starting Enhanced Blender Bridge monitoring...")
    print(f"Watching directory: {COMMUNICATION_DIR}")
    print("Waiting for character generation requests...")
   
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
                print("Error: Received request file missing 'structured_data'.")
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

            print(f"🎭 Received enhanced request for prompt: {prompt}")
            print(f"📊 Properties to apply: {len(structured_data.get('properties', {}))}")
           
            # For now, use male character by default (you can enhance this with gender detection)
            character = get_object("mb_male")
           
            if character:
                # Use the new enhanced processing function
                process_enhanced_properties(structured_data, character)
               
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "status": "completed",
                    "message": f"Character generated with {len(structured_data.get('properties', {}))} properties!",
                    "properties_applied": len(structured_data.get('properties', {}))
                }
            else:
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "status": "error",
                    "message": "Could not find character object in Blender scene."
                }
           
            # Send response
            with open(RESPONSE_FILE, 'w') as f:
                json.dump(response_data, f)
           
            # Remove request file
            os.remove(REQUEST_FILE)
           
    except Exception as e:
        print(f"Error processing request: {e}")
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
        row = layout.row()
        row.operator("mesh.start_bridge")
        row = layout.row()
        row.operator("mesh.stop_bridge")
        row = layout.row()
        row.operator("mesh.test_enhanced_generation")

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
    """Test enhanced character generation with sample properties."""
    bl_idname = "mesh.test_enhanced_generation"
    bl_label = "Test Enhanced Generation"
    
    def execute(self, context):
        # Sample enhanced property mapping (simulating NLP+LLM output)
        test_structured_data = {
            "properties": {
                "L2__Eyes_Size_max": 0.8,
                "L2_Caucasian_Nose_TipSize_min": 0.7,
                "L2_Caucasian_Jaw_Angle_min": 0.9,
                "L2__Body_Size_max": 0.6,
                "L1_Asian": 0.8
            },
            "analysis": {
                "analysis": "Test: Athletic Asian male with sharp features",
                "cultural_context": "Detected cultural indicators: asian",
                "lifestyle_traits": "Detected lifestyle: athletic"
            }
        }
        
        character = get_object("mb_male")
        
        if character:
            process_enhanced_properties(test_structured_data, character)
            self.report({'INFO'}, "Enhanced test generation completed!")
        else:
            self.report({'ERROR'}, "Character object not found")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(MESH_PT_character_bridge)
    bpy.utils.register_class(MESH_OT_start_bridge)
    bpy.utils.register_class(MESH_OT_stop_bridge)
    bpy.utils.register_class(MESH_OT_test_enhanced_generation)

def unregister():
    bpy.utils.unregister_class(MESH_PT_character_bridge)
    bpy.utils.unregister_class(MESH_OT_start_bridge)
    bpy.utils.unregister_class(MESH_OT_stop_bridge)
    bpy.utils.unregister_class(MESH_OT_test_enhanced_generation)

# Register classes
register()

# =============================================================================
# AUTO-START AND MAIN EXECUTION
# =============================================================================
print("=== ENHANCED CHARACTER GENERATOR BRIDGE (NLP+LLM) LOADED ===")
print("Features:")
print("  ✓ NLP-based property mapping")
print("  ✓ LLM-enhanced analysis") 
print("  ✓ Cultural context detection")
print("  ✓ Lifestyle trait mapping")
print()
print("Instructions:")
print("1. Run the enhanced frontend.py script")
print("2. Click 'Start Bridge' in Blender")
print("3. Open http://127.0.0.1:5000 and generate characters!")

# Optional: Uncomment to auto-start monitoring
# start_bridge_monitoring()

if __name__ == "__main__":
    # Test with sample enhanced data
    mock_data = {
        "properties": {
            "L2__Eyes_Size_max": 0.8,
            "L2_Caucasian_Mouth_UpperlipVolume_max": 0.7,
            "L2__Body_Size_min": 0.6
        },
        "analysis": {
            "analysis": "Test character with enhanced properties"
        }
    }
    
    character = get_object("mb_male")
    if character:
        process_enhanced_properties(mock_data, character)
        print("✓ Enhanced test completed successfully")
    else:
        print("✗ Character object not found for testing")