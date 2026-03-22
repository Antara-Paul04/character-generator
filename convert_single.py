"""Convert a single .obj file to .blend - run inside Blender background mode"""
import os
import sys
import bpy

# Get the obj path from command line args after "--"
argv = sys.argv
args = argv[argv.index("--") + 1:]
obj_path = args[0]
blend_path = args[1]

name = os.path.splitext(os.path.basename(obj_path))[0]

try:
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)

    # Find mesh
    mesh_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break

    if not mesh_obj:
        print(f"FAIL: No mesh found in {name}")
        sys.exit(1)

    mesh_obj.name = name
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.shade_smooth()

    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"OK: {name}")
except Exception as e:
    print(f"FAIL: {name}: {e}")
    sys.exit(1)
