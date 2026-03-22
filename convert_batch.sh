#!/bin/bash
# Convert remaining .obj files to .blend one at a time
BLENDER="/c/Program Files/Blender Foundation/Blender 4.5/blender.exe"
SCRIPT="C:/Users/paula/char-gen/convert_single.py"
OUTDIR="C:/Users/paula/char-gen/clothing_assets"
DLDIR="C:/Users/paula/char-gen/mh_clothing_raw"

mkdir -p "$OUTDIR"

converted=0
skipped=0
failed=0

find "$DLDIR" -name "*.obj" -type f | sort | while read objfile; do
    name=$(basename "$objfile" .obj)
    blend="$OUTDIR/${name}.blend"

    if [ -f "$blend" ]; then
        echo "SKIP: $name (already exists)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "Converting: $name..."
    "$BLENDER" --background --python "$SCRIPT" -- "$objfile" "$blend" 2>/dev/null

    if [ -f "$blend" ]; then
        echo "  OK: $name"
        converted=$((converted + 1))
    else
        echo "  FAILED: $name"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Done! Check clothing_assets/ for results."
