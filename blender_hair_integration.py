# This file should be placed in your Blender scripts folder or alongside blender_bridge.py
import bpy
import os
from mathutils import Vector

def import_hair_obj(obj_path: str, character_obj) -> bool:
    """
    Import hair .obj file and attach it to character
    """
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
        
        # Find the character's head location for positioning
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

def get_head_location(character_obj) -> Vector:
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

def setup_hair_material(hair_objects: list):
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

def create_particle_hair_system(character_obj, hair_params: dict):
    """Create a Blender particle hair system"""
    try:
        bpy.context.view_layer.objects.active = character_obj
        character_obj.select_set(True)
        
        bpy.ops.object.particle_system_add()
        
        psys = character_obj.particle_systems[-1]
        psys.name = "Hair_System"
        
        settings = psys.settings
        settings.type = 'HAIR'
        
        # Apply parameters from hair_params
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

def apply_hair_to_character(character_obj, hair_data: dict) -> dict:
    """Main function to apply hair to character - called from blender_bridge.py"""
    result = {
        'success': False,
        'method': None,
        'message': ''
    }
    
    # Method 1: Try to import HairNet .obj file
    if 'output_path' in hair_data and not hair_data.get('mock', False):
        obj_path = hair_data['output_path']
        if import_hair_obj(obj_path, character_obj):
            result['success'] = True
            result['method'] = 'hairnet_obj'
            result['message'] = 'HairNet OBJ imported successfully'
            return result
    
    # Method 2: Fallback to particle system
    if 'hair_params' in hair_data:
        if create_particle_hair_system(character_obj, hair_data['hair_params']):
            result['success'] = True
            result['method'] = 'particle_system'
            result['message'] = 'Particle hair system created'
            return result
    
    result['message'] = 'All hair application methods failed'
    return result