"""
MakeHuman Clothing Asset Downloader & Converter
Downloads MakeHuman community clothing packs (.obj) and converts them to .blend files
for use with CharMorph fitting in Blender.

Usage:
    Step 1 (run from terminal):  python clothing_setup.py download
    Step 2 (run inside Blender): Open Blender > Scripting tab > Open this file > Run Script
                                  OR: blender --background --python clothing_setup.py -- convert
"""

import os
import sys
import zipfile
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mh_clothing_raw")
CONVERTED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clothing_assets")

# CharMorph asset directories
CHARMORPH_FEMALE_ASSETS = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Roaming", "Blender Foundation", "Blender", "4.5",
    "scripts", "addons", "CharMorph", "data", "characters", "MB-Lab Female", "assets"
)
CHARMORPH_MALE_ASSETS = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Roaming", "Blender Foundation", "Blender", "4.5",
    "scripts", "addons", "CharMorph", "data", "characters", "MB-Lab Male", "assets"
)

# MakeHuman clothing asset packs (CC0 licensed)
ASSET_PACKS = {
    "shirts": {
        "url": "https://files2.makehumancommunity.org/asset_packs/shirts01/shirts01_cc0.zip",
        "size": "23 MB",
        "description": "T-shirts, sweaters, tops",
    },
    "pants": {
        "url": "https://files2.makehumancommunity.org/asset_packs/pants01/pants01_cc0.zip",
        "size": "20 MB",
        "description": "Cargo pants, jeans shorts, harem pants, wool pants",
    },
    "dress": {
        "url": "https://files2.makehumancommunity.org/asset_packs/dress01/dress01_cc0.zip",
        "size": "44 MB",
        "description": "14 dresses - flapper, kimono, halter, tunics",
    },
    "skirts": {
        "url": "https://files2.makehumancommunity.org/asset_packs/skirts01/skirts01_cc0.zip",
        "size": "28 MB",
        "description": "6 skirts",
    },
    "suits": {
        "url": "https://files2.makehumancommunity.org/asset_packs/suits01/suits01_cc0.zip",
        "size": "40 MB",
        "description": "Formal suits",
    },
    "shoes": {
        "url": "https://files2.makehumancommunity.org/asset_packs/shoes01/shoes01_cc0.zip",
        "size": "79 MB",
        "description": "24 shoes and boots",
    },
    "underwear": {
        "url": "https://files2.makehumancommunity.org/asset_packs/underwear01/underwear01_cc0.zip",
        "size": "57 MB",
        "description": "Underwear collection",
    },
}


# =============================================================================
# STEP 1: DOWNLOAD (run with regular Python)
# =============================================================================

def download_packs(packs=None):
    """Download MakeHuman clothing asset packs"""
    import urllib.request

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if packs is None:
        packs = list(ASSET_PACKS.keys())

    for pack_name in packs:
        pack = ASSET_PACKS.get(pack_name)
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
                # Try mirror 2
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

        # Extract
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
                obj_path = os.path.join(root, f)
                obj_files.append(obj_path)
    return obj_files


def list_downloaded():
    """List all downloaded .obj files"""
    if not os.path.exists(DOWNLOAD_DIR):
        print("No downloads yet. Run: python clothing_setup.py download")
        return

    obj_files = find_obj_files(DOWNLOAD_DIR)
    print(f"\nFound {len(obj_files)} .obj clothing files:\n")
    for f in sorted(obj_files):
        name = os.path.splitext(os.path.basename(f))[0]
        size_kb = os.path.getsize(f) / 1024
        print(f"  {name:40s} ({size_kb:.0f} KB)")


# =============================================================================
# STEP 2: CONVERT (run inside Blender)
# =============================================================================

def convert_obj_to_blend():
    """
    Convert all downloaded .obj files to .blend files.
    MUST be run inside Blender (uses bpy).
    """
    try:
        import bpy
    except ImportError:
        print("ERROR: This function must be run inside Blender!")
        print("  Option 1: Open Blender > Scripting > Open this file > Run")
        print("  Option 2: blender --background --python clothing_setup.py -- convert")
        return

    os.makedirs(CONVERTED_DIR, exist_ok=True)

    obj_files = find_obj_files(DOWNLOAD_DIR)
    if not obj_files:
        print("No .obj files found. Run 'python clothing_setup.py download' first.")
        return

    print(f"\nConverting {len(obj_files)} .obj files to .blend...")
    converted = 0
    failed = 0

    for obj_path in sorted(obj_files):
        name = os.path.splitext(os.path.basename(obj_path))[0]
        blend_path = os.path.join(CONVERTED_DIR, f"{name}.blend")

        if os.path.exists(blend_path):
            print(f"  [{name}] Already converted, skipping")
            converted += 1
            continue

        print(f"  [{name}] Converting...")

        try:
            # Clear the scene
            bpy.ops.wm.read_factory_settings(use_empty=True)

            # Import the OBJ file
            try:
                # Blender 4.x uses new importer
                bpy.ops.wm.obj_import(filepath=obj_path)
            except AttributeError:
                # Fallback for older Blender
                bpy.ops.import_scene.obj(filepath=obj_path)

            # Find the imported mesh object
            mesh_obj = None
            for obj in bpy.context.scene.objects:
                if obj.type == 'MESH':
                    mesh_obj = obj
                    break

            if not mesh_obj:
                print(f"  [{name}] WARNING: No mesh found after import, skipping")
                failed += 1
                continue

            # Rename the object to match the file name
            mesh_obj.name = name

            # Apply transforms
            bpy.context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Smooth shading
            bpy.ops.object.shade_smooth()

            # Save as .blend
            bpy.ops.wm.save_as_mainfile(filepath=blend_path)
            converted += 1
            print(f"  [{name}] OK -> {os.path.basename(blend_path)}")

        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            failed += 1

    print(f"\nConversion complete: {converted} OK, {failed} failed")
    print(f"Blend files saved to: {CONVERTED_DIR}")


# =============================================================================
# STEP 3: INSTALL (copy to CharMorph assets folder)
# =============================================================================

def install_to_charmorph():
    """Copy converted .blend files to CharMorph asset directories"""
    if not os.path.exists(CONVERTED_DIR):
        print("No converted files. Run the converter first.")
        return

    blend_files = [f for f in os.listdir(CONVERTED_DIR) if f.endswith('.blend')]
    if not blend_files:
        print("No .blend files found in converted directory.")
        return

    # Install to female assets (MakeHuman clothes are generally unisex/female)
    target_dir = CHARMORPH_FEMALE_ASSETS
    if not os.path.exists(target_dir):
        print(f"CharMorph female assets dir not found: {target_dir}")
        print("Make sure CharMorph addon is installed.")
        return

    installed = 0
    for blend_file in sorted(blend_files):
        src = os.path.join(CONVERTED_DIR, blend_file)
        dst = os.path.join(target_dir, blend_file)
        if os.path.exists(dst):
            print(f"  [{blend_file}] Already installed, skipping")
            installed += 1
            continue

        shutil.copy2(src, dst)
        installed += 1
        print(f"  [{blend_file}] Installed to CharMorph")

    print(f"\nInstalled {installed} clothing assets to CharMorph")
    print(f"Location: {target_dir}")


# =============================================================================
# STEP 4: UPDATE blender_bridge.py asset map
# =============================================================================

def generate_asset_map():
    """Generate an updated CLOTHING_ASSET_MAP from installed .blend files"""
    if not os.path.exists(CONVERTED_DIR):
        print("No converted files yet.")
        return

    blend_files = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(CONVERTED_DIR)
        if f.endswith('.blend')
    ])

    print(f"\n# Available clothing assets ({len(blend_files)} items):")
    print("# Add these to CLOTHING_ASSET_MAP in blender_bridge.py\n")

    # Auto-categorize based on filename
    categories = {
        "tops": [], "bottoms": [], "dresses": [],
        "shoes": [], "underwear": [], "suits": [], "other": []
    }

    top_words = ["shirt", "top", "blouse", "sweater", "jacket", "vest", "tank",
                 "tshirt", "polo", "hoodie", "coat", "cardigan", "tunic"]
    bottom_words = ["pants", "trousers", "jeans", "shorts", "skirt", "legging"]
    dress_words = ["dress", "gown", "kimono", "robe", "frock"]
    shoe_words = ["shoe", "boot", "sandal", "sneaker", "heel", "slipper", "loafer"]
    underwear_words = ["bra", "panty", "panties", "underwear", "brief", "boxer", "thong",
                       "bikini", "lingerie", "corset"]
    suit_words = ["suit", "tuxedo", "blazer", "formal"]

    for name in blend_files:
        name_lower = name.lower()
        if any(w in name_lower for w in dress_words):
            categories["dresses"].append(name)
        elif any(w in name_lower for w in top_words):
            categories["tops"].append(name)
        elif any(w in name_lower for w in bottom_words):
            categories["bottoms"].append(name)
        elif any(w in name_lower for w in shoe_words):
            categories["shoes"].append(name)
        elif any(w in name_lower for w in underwear_words):
            categories["underwear"].append(name)
        elif any(w in name_lower for w in suit_words):
            categories["suits"].append(name)
        else:
            categories["other"].append(name)

    for cat, items in categories.items():
        if items:
            print(f"  # {cat.upper()} ({len(items)})")
            for item in items:
                print(f'  "{item}": "{item}",')
            print()


# =============================================================================
# CLI
# =============================================================================

def print_usage():
    print("""
MakeHuman Clothing Asset Setup
==============================

Usage:
  python clothing_setup.py download          Download all clothing packs (~250 MB)
  python clothing_setup.py download shirts   Download only shirts pack
  python clothing_setup.py list              List downloaded .obj files
  python clothing_setup.py install           Copy .blend files to CharMorph
  python clothing_setup.py map               Generate asset map for blender_bridge.py

  (Inside Blender or with --background):
  blender --background --python clothing_setup.py -- convert

Available packs: shirts, pants, dress, skirts, suits, shoes, underwear

Full setup:
  1. python clothing_setup.py download
  2. blender --background --python clothing_setup.py -- convert
  3. python clothing_setup.py install
  4. python clothing_setup.py map
""")


if __name__ == "__main__":
    # Handle Blender's "--" argument separator
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
        print("Downloading MakeHuman clothing packs...\n")
        download_packs(packs)
    elif cmd == "list":
        list_downloaded()
    elif cmd == "convert":
        convert_obj_to_blend()
    elif cmd == "install":
        install_to_charmorph()
    elif cmd == "map":
        generate_asset_map()
    else:
        print(f"Unknown command: {cmd}")
        print_usage()
