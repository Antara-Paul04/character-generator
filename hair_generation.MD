# Hair Generation Setup

## Prerequisites
1. Blender 4.2+ (required for GBH Tool v2.2.x)
2. GBH Tool addon installed in Blender
3. Python 3.9+ for the web backend

## Installation Steps

### 1. Install GBH Tool in Blender
1. Download GBH Tool from https://github.com/GixoXYZ/BlenderGBHTool
2. In Blender: Edit → Preferences → Get Extensions
3. Click dropdown menu → Install from Disk
4. Select the downloaded GBH Tool zip file
5. Enable the addon (search for "GBH" in addons list)

### 2. Verify GBH Tool Installation
Run this in Blender's Python console:
```python
import bpy
print([op for op in dir(bpy.ops) if 'gbh' in op.lower()])