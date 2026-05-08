"""
MakeHuman Hair Asset Downloader & Converter
Downloads MakeHuman community hair packs (.obj) and converts them to .blend files.

Usage:
    Step 1: python hair_setup.py download
    Step 2: blender --background --python hair_setup.py -- convert
    Step 3: python hair_setup.py list
"""

import os
import sys
import zipfile

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mh_hair_raw")
CONVERTED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hair_assets")

HAIR_PACKS = {
    "hair01": {
        "url": "https://files2.makehumancommunity.org/asset_packs/hair01/hair01_cc0.zip",
        "size": "217 MB",
        "description": "26 hairstyles - bobs, braids, buns, anime, shaggy (CC0)",
    },
    "hair02": {
        "url": "https://files2.makehumancommunity.org/asset_packs/hair02/hair02_ccby.zip",
        "size": "68 MB",
        "description": "23 hairstyles - afro, updo, wavy bob, braided rows (CC-BY)",
    },
    "hair03": {
        "url": "https://files2.makehumancommunity.org/asset_packs/hair03/hair03_ccby.zip",
        "size": "38 MB",
        "description": "14 hairstyles - chinese bob, curly, long, bun with braids (CC-BY)",
    },
}


def download_packs(packs=None):
    """Download MakeHuman hair asset packs"""
    import urllib.request

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if packs is None:
        packs = list(HAIR_PACKS.keys())

    for pack_name in packs:
        pack = HAIR_PACKS.get(pack_name)
        if not pack:
            print(f"  Unknown pack: {pack_name}")
            continue

        zip_path = os.path.join(DOWNLOAD_DIR, f"{pack_name}.zip")
        extract_dir = os.path.join(DOWNLOAD_DIR, pack_name)

        if os.path.exists(extract_dir):
            print(f"  [{pack_name}] Already extracted, skipping download")
            continue

        if not os.path.exists(zip_path):
            print(f"  [{pack_name}] Downloading {pack['size']}... ({pack['description']})")
            try:
                urllib.request.urlretrieve(pack["url"], zip_path)
                print(f"  [{pack_name}] Downloaded OK")
            except Exception as e:
                print(f"  [{pack_name}] Download FAILED: {e}")
                mirror2 = pack["url"].replace("files2.", "files.")
                try:
                    print(f"  [{pack_name}] Trying mirror 2...")
                    urllib.request.urlretrieve(mirror2, zip_path)
                    print(f"  [{pack_name}] Downloaded OK (mirror 2)")
                except Exception as e2:
                    print(f"  [{pack_name}] Mirror 2 also failed: {e2}")
                    continue
        else:
            print(f"  [{pack_name}] Zip already exists")

        print(f"  [{pack_name}] Extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            print(f"  [{pack_name}] Extracted OK")
        except Exception as e:
            print(f"  [{pack_name}] Extract FAILED: {e}")


def find_obj_files(base_dir):
    """Find all .obj files in the downloaded assets"""
    obj_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, f))
    return obj_files


def list_downloaded():
    """List all downloaded .obj files"""
    if not os.path.exists(DOWNLOAD_DIR):
        print("No downloads yet. Run: python hair_setup.py download")
        return

    obj_files = find_obj_files(DOWNLOAD_DIR)
    print(f"\nFound {len(obj_files)} .obj hair files:\n")
    for f in sorted(obj_files):
        name = os.path.splitext(os.path.basename(f))[0]
        size_kb = os.path.getsize(f) / 1024
        print(f"  {name:40s} ({size_kb:.0f} KB)")


def list_converted():
    """List converted .blend files"""
    if not os.path.exists(CONVERTED_DIR):
        print("No converted files yet.")
        return

    blend_files = sorted([f for f in os.listdir(CONVERTED_DIR) if f.endswith('.blend')])
    print(f"\nFound {len(blend_files)} converted hair .blend files:\n")
    for f in blend_files:
        size_kb = os.path.getsize(os.path.join(CONVERTED_DIR, f)) / 1024
        print(f"  {f:40s} ({size_kb:.0f} KB)")


def print_usage():
    print("""
MakeHuman Hair Asset Setup
==========================

Usage:
  python hair_setup.py download          Download all hair packs (~323 MB)
  python hair_setup.py download hair01   Download only hair01 pack
  python hair_setup.py list              List downloaded .obj files
  python hair_setup.py converted         List converted .blend files

  (Inside Blender or with --background):
  blender --background --python hair_setup.py -- convert

Available packs: hair01 (26 styles), hair02 (23 styles), hair03 (14 styles)

Full setup:
  1. python hair_setup.py download
  2. Use convert_hair_batch.sh to convert one at a time
""")


if __name__ == "__main__":
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    else:
        args = sys.argv[1:]

    if not args:
        print_usage()
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "download":
        packs = args[1:] if len(args) > 1 else None
        print("Downloading MakeHuman hair packs...\n")
        download_packs(packs)
    elif cmd == "list":
        list_downloaded()
    elif cmd == "converted":
        list_converted()
    elif cmd == "convert":
        # This runs inside Blender
        try:
            import bpy
        except ImportError:
            print("ERROR: 'convert' must be run inside Blender!")
            print("Use: blender --background --python hair_setup.py -- convert")
            sys.exit(1)

        os.makedirs(CONVERTED_DIR, exist_ok=True)
        obj_files = find_obj_files(DOWNLOAD_DIR)
        if not obj_files:
            print("No .obj files found. Run 'python hair_setup.py download' first.")
            sys.exit(1)

        print(f"\nConverting {len(obj_files)} hair .obj files...")
        for obj_path in sorted(obj_files):
            name = os.path.splitext(os.path.basename(obj_path))[0]
            blend_path = os.path.join(CONVERTED_DIR, f"{name}.blend")
            if os.path.exists(blend_path):
                print(f"  [{name}] Already converted")
                continue
            print(f"  [{name}] Converting...")
            try:
                bpy.ops.wm.read_factory_settings(use_empty=True)
                bpy.ops.wm.obj_import(filepath=obj_path)
                mesh_obj = None
                for obj in bpy.context.scene.objects:
                    if obj.type == 'MESH':
                        mesh_obj = obj
                        break
                if not mesh_obj:
                    print(f"  [{name}] No mesh found, skipping")
                    continue
                mesh_obj.name = name
                bpy.context.view_layer.objects.active = mesh_obj
                mesh_obj.select_set(True)
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                bpy.ops.object.shade_smooth()
                bpy.ops.wm.save_as_mainfile(filepath=blend_path)
                print(f"  [{name}] OK")
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")

        print("\nConversion complete!")
    else:
        print(f"Unknown command: {cmd}")
        print_usage()
