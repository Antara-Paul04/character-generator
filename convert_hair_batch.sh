#!/bin/bash
BLENDER="/c/Program Files/Blender Foundation/Blender 4.5/blender.exe"
SCRIPT="C:/Users/paula/char-gen/convert_single.py"
OUTDIR="C:/Users/paula/char-gen/hair_assets"
DLDIR="C:/Users/paula/char-gen/mh_hair_raw"

mkdir -p "$OUTDIR"

find "$DLDIR" -name "*.obj" -type f | sort | while read objfile; do
    name=$(basename "$objfile" .obj)
    blend="$OUTDIR/${name}.blend"

    if [ -f "$blend" ]; then
        echo "SKIP: $name"
        continue
    fi

    echo "Converting: $name..."
    "$BLENDER" --background --python "$SCRIPT" -- "$objfile" "$blend" 2>/dev/null

    if [ -f "$blend" ]; then
        echo "  OK: $name"
    else
        echo "  FAILED: $name"
    fi
done

echo ""
echo "Done! Check hair_assets/ for results."
