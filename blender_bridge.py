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
# HAIR GENERATION SYSTEM (GBH Tool Integration)
# =============================================================================

class HairGenerator:
    """Integrates GBH Tool for procedural hair generation"""
    
    def __init__(self):
        self.hair_styles = {
            'short': {
                'length': 0.05,
                'subdivisions': 2,
                'children': 50,
                'clump': 0.0,
                'roughness': 0.1
            },
            'medium': {
                'length': 0.15,
                'subdivisions': 3,
                'children': 100,
                'clump': 0.1,
                'roughness': 0.15
            },
            'long': {
                'length': 0.35,
                'subdivisions': 4,
                'children': 150,
                'clump': 0.2,
                'roughness': 0.2
            },
            'very_long': {
                'length': 0.6,
                'subdivisions': 5,
                'children': 200,
                'clump': 0.25,
                'roughness': 0.25
            }
        }
        
        self.hair_types = {
            'straight': {'roughness': 0.0, 'clump': 0.0},
            'wavy': {'roughness': 0.15, 'clump': 0.1},
            'curly': {'roughness': 0.35, 'clump': 0.3},
            'kinky': {'roughness': 0.5, 'clump': 0.4}
        }
        
        self.hairstyles = {
            'ponytail': 'ponytail',
            'bun': 'bun',
            'braids': 'braids',
            'bob': 'bob',
            'pixie': 'short',
            'buzz': 'very_short',
            'afro': 'afro'
        }
    
    def detect_hair_features(self, prompt: str) -> Dict:
        """Detect hair characteristics from prompt"""
        prompt_lower = prompt.lower()
        
        features = {
            'generate_hair': False,
            'length': 'medium',
            'type': 'straight',
            'style': None,
            'color': (0.05, 0.02, 0.01),  # Default dark brown
            'density': 1.0
        }
        
        # Check if hair is mentioned
        hair_keywords = ['hair', 'hairstyle', 'haircut', 'locks', 'mane', 'tresses']
        if any(keyword in prompt_lower for keyword in hair_keywords):
            features['generate_hair'] = True
        
        # Detect length
        if any(word in prompt_lower for word in ['long hair', 'flowing hair', 'lengthy']):
            features['length'] = 'long'
        elif any(word in prompt_lower for word in ['very long', 'extremely long', 'waist-length']):
            features['length'] = 'very_long'
        elif any(word in prompt_lower for word in ['short hair', 'cropped', 'pixie']):
            features['length'] = 'short'
        elif any(word in prompt_lower for word in ['medium hair', 'shoulder-length']):
            features['length'] = 'medium'
        
        # Detect type
        if any(word in prompt_lower for word in ['straight hair', 'silky', 'sleek']):
            features['type'] = 'straight'
        elif any(word in prompt_lower for word in ['wavy hair', 'waves']):
            features['type'] = 'wavy'
        elif any(word in prompt_lower for word in ['curly hair', 'curls', 'ringlets']):
            features['type'] = 'curly'
        elif any(word in prompt_lower for word in ['kinky', 'coily', 'afro textured']):
            features['type'] = 'kinky'
        
        # Detect specific styles
        for style_name, style_key in self.hairstyles.items():
            if style_name in prompt_lower:
                features['style'] = style_key
                break
        
        # Detect color
        color_map = {
            'black': (0.01, 0.01, 0.01),
            'dark brown': (0.05, 0.02, 0.01),
            'brown': (0.15, 0.08, 0.04),
            'light brown': (0.25, 0.15, 0.08),
            'blonde': (0.8, 0.7, 0.4),
            'red': (0.4, 0.1, 0.05),
            'auburn': (0.3, 0.08, 0.03),
            'white': (0.9, 0.9, 0.9),
            'gray': (0.5, 0.5, 0.5),
            'grey': (0.5, 0.5, 0.5)
        }
        
        for color_name, rgb in color_map.items():
            if f'{color_name} hair' in prompt_lower or f'{color_name}hair' in prompt_lower:
                features['color'] = rgb
                break
        
        # Detect density
        if any(word in prompt_lower for word in ['thick hair', 'full hair', 'voluminous']):
            features['density'] = 1.5
        elif any(word in prompt_lower for word in ['thin hair', 'fine hair', 'sparse']):
            features['density'] = 0.7
        
        return features
    
    def generate_hair_system(self, character_obj, hair_features: Dict) -> bool:
        """Generate hair system using particle system"""
        try:
            # Select character
            bpy.ops.object.select_all(action='DESELECT')
            character_obj.select_set(True)
            bpy.context.view_layer.objects.active = character_obj
            
            # Remove existing hair systems
            for modifier in character_obj.modifiers:
                if modifier.type == 'PARTICLE_SYSTEM':
                    character_obj.modifiers.remove(modifier)
            
            # Clear particle systems
            character_obj.particle_systems.clear()
            
            # Add new particle system
            bpy.ops.object.particle_system_add()
            particle_system = character_obj.particle_systems[-1]
            settings = particle_system.settings
            
            # Get style parameters
            length_params = self.hair_styles.get(hair_features['length'], self.hair_styles['medium'])
            type_params = self.hair_types.get(hair_features['type'], self.hair_types['straight'])
            
            # Configure particle system as hair
            settings.type = 'HAIR'
            settings.count = int(1000 * hair_features['density'])
            settings.hair_length = length_params['length']
            settings.path_end = 1.0
            
            # Subdivisions for smoother hair
            settings.render_step = length_params['subdivisions']
            settings.display_step = length_params['subdivisions']
            
            # Hair dynamics
            settings.use_advanced_hair = True
            settings.clump_factor = max(length_params['clump'], type_params['clump'])
            settings.roughness_1 = type_params['roughness']
            settings.roughness_1_size = 0.5
            settings.roughness_endpoint = 0.3
            
            # Children particles for volume
            settings.child_type = 'INTERPOLATED'
            settings.child_nbr = length_params['children']
            settings.rendered_child_count = length_params['children'] * 2
            settings.clump_factor = type_params['clump']
            settings.clump_shape = 0.0
            
            # Set hair material
            self.create_hair_material(character_obj, hair_features['color'])
            
            print(f"✓ Generated {hair_features['length']} {hair_features['type']} hair")
            return True
            
        except Exception as e:
            print(f"✗ Hair generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_hair_material(self, obj, color: Tuple[float, float, float]):
        """Create or update hair material"""
        mat_name = "Hair_Material"
        
        # Get or create material
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
        
        # Clear existing nodes
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Create nodes
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        
        # Set hair color
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.4
        bsdf.inputs['Sheen Tint'].default_value = 0.5
        
        # Connect nodes
        links = mat.node_tree.links
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Assign material to particle system
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        
        print(f"✓ Created hair material with color RGB{color}")

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
    
    print(f"✗ Could not find any character object")
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
# ENHANCED PROPERTY PROCESSING WITH HAIR
# =============================================================================
def process_enhanced_properties(structured_data: Dict, character_obj, gender: str, prompt: str):
    """
    Applies morphs and generates hair from enhanced property mapping.
    """
    print(f"--- Starting Enhanced Property Processing for {gender.upper()} character ---")
    reset_character_shape_keys(character_obj)
    
    properties = structured_data.get("properties", {})
    analysis = structured_data.get("analysis", {})
    
    print(f"Processing {len(properties)} properties from analysis...")
    
    # Apply morphs
    applied_count = 0
    failed_count = 0
    
    for property_name, intensity in properties.items():
        if apply_morph(character_obj, property_name, intensity):
            applied_count += 1
        else:
            failed_count += 1
    
    # Generate hair if mentioned in prompt
    hair_generator = HairGenerator()
    hair_features = hair_generator.detect_hair_features(prompt)
    
    hair_generated = False
    if hair_features['generate_hair']:
        print(f"\n--- Hair Generation Detected ---")
        print(f"  Length: {hair_features['length']}")
        print(f"  Type: {hair_features['type']}")
        print(f"  Style: {hair_features.get('style', 'default')}")
        print(f"  Color: RGB{hair_features['color']}")
        print(f"  Density: {hair_features['density']}")
        
        hair_generated = hair_generator.generate_hair_system(character_obj, hair_features)
    
    bpy.context.view_layer.update()
    
    print(f"\n✓ Enhanced character generation complete!")
    print(f"  - Applied {applied_count} morphs successfully")
    if failed_count > 0:
        print(f"  - Failed to apply {failed_count} morphs")
    if hair_generated:
        print(f"  - Hair system generated successfully")
    
    # Print analysis summary
    if analysis:
        print("\n📊 Analysis Summary:")
        if analysis.get('analysis'):
            print(f"  - {analysis['analysis']}")
        if analysis.get('cultural_context'):
            print(f"  - {analysis['cultural_context']}")
        if analysis.get('lifestyle_traits'):
            print(f"  - {analysis['lifestyle_traits']}")
    
    return applied_count, failed_count, hair_generated

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
   
    print(f"🎭 Starting Enhanced Blender Bridge with Hair Generation...")
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
            
            print(f"\n{'='*60}")
            print(f"🎭 NEW CHARACTER REQUEST")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"Gender: {gender.upper()}")
            print(f"Ethnicity: {ethnicity.upper()}")
            print(f"Properties to apply: {len(structured_data.get('properties', {}))}")
            print(f"{'='*60}\n")
           
            character = get_object_by_gender(gender)
           
            if character:
                applied, failed, hair_gen = process_enhanced_properties(
                    structured_data, character, gender, prompt
                )
               
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "status": "completed",
                    "message": f"✓ {gender.capitalize()} character with hair generated!",
                    "properties_applied": applied,
                    "properties_failed": failed,
                    "hair_generated": hair_gen,
                    "character_object": character.name
                }
            else:
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "gender": gender,
                    "status": "error",
                    "message": f"Could not find {gender} character object."
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

# =============================================================================
# BLENDER UI PANEL
# =============================================================================
class MESH_PT_character_bridge(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Enhanced Character Generator with Hair"
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
        row.operator("mesh.test_with_hair")

class MESH_OT_start_bridge(bpy.types.Operator):
    """Start the bridge monitoring."""
    bl_idname = "mesh.start_bridge"
    bl_label = "Start Bridge"
    
    def execute(self, context):
        start_bridge_monitoring()
        self.report({'INFO'}, "Enhanced bridge with hair generation started")
        return {'FINISHED'}

class MESH_OT_stop_bridge(bpy.types.Operator):
    """Stop the bridge monitoring."""
    bl_idname = "mesh.stop_bridge"
    bl_label = "Stop Bridge"
    
    def execute(self, context):
        stop_bridge_monitoring()
        self.report({'INFO'}, "Bridge monitoring stopped")
        return {'FINISHED'}

class MESH_OT_test_with_hair(bpy.types.Operator):
    """Test character generation with hair."""
    bl_idname = "mesh.test_with_hair"
    bl_label = "Test with Hair"
    
    def execute(self, context):
        test_prompt = "Create a young woman with long wavy brown hair, big eyes, and a warm smile."
        test_structured_data = {
            "properties": {
                "L1_Caucasian": 0.8,
                "L2__Eyes_Size_max": 0.8,
                "L2_Caucasian_Mouth_UpperlipVolume_max": 0.7,
            },
            "analysis": {"analysis": "Test character with hair"},
            "gender": "female",
            "ethnicity": "caucasian"
        }
        
        character = get_object_by_gender("female")
        
        if character:
            process_enhanced_properties(test_structured_data, character, "female", test_prompt)
            self.report({'INFO'}, "Test generation with hair completed!")
        else:
            self.report({'ERROR'}, "Character object not found")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(MESH_PT_character_bridge)
    bpy.utils.register_class(MESH_OT_start_bridge)
    bpy.utils.register_class(MESH_OT_stop_bridge)
    bpy.utils.register_class(MESH_OT_test_with_hair)

def unregister():
    bpy.utils.unregister_class(MESH_PT_character_bridge)
    bpy.utils.unregister_class(MESH_OT_start_bridge)
    bpy.utils.unregister_class(MESH_OT_stop_bridge)
    bpy.utils.unregister_class(MESH_OT_test_with_hair)

register()

print("="*70)
print("🎭 ENHANCED CHARACTER GENERATOR WITH HAIR SYSTEM")
print("="*70)
print("Features:")
print("  ✓ Automatic gender detection (male/female)")
print("  ✓ Procedural hair generation")
print("  ✓ Hair length, type, color, and style detection")
print("  ✓ 30+ facial/body properties")
print("  ✓ NLP + LLM enhanced analysis")
print("="*70)