#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

ID=$1
TARGET_DIR="backup_$ID"
TARGETS=("condinit_drift_z0" "condinit_drift_z0p5" "condinit_drift_z1" "condinit_nodrift_z0" "condinit_nodrift_z0p5" "condinit_nodrift_z1")

for dir in "${TARGETS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Warning: Directory '$dir' already exists. Move it to backup directory or delete itbefore restoring..."
        exit 1
    fi
done

if [ ! -d "backups/$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

echo "=============================================="
echo "Restoring from backup: $TARGET_DIR"
echo "=============================================="

for subfolder in "${TARGETS[@]}"; do
    if [ -d "backups/$TARGET_DIR/$subfolder" ]; then
        echo "Restoring '$subfolder'..."
        mv "backups/$TARGET_DIR/$subfolder" ./
    else
        echo "Warning: '$subfolder' not found in backup. Skipping."
    fi
done

rmdir "backups/$TARGET_DIR" 2>/dev/null || rm -rf "backups/$TARGET_DIR"

echo "==============================================="
echo "Restoration completed from: $TARGET_DIR"
echo "==============================================="

cd backups || exit

folders=$(ls -d backup_[0-9][0-9] 2>/dev/null | sort)

count=0
for folder in $folders; do
    new_name=$(printf "backup_%02d" $count)

    if [ "$folder" != "$new_name" ]; then
        echo "Renaming '$folder' to '$new_name'..."
        mv "$folder" "$new_name"
    fi
    ((count++))
done

echo "=============================================="
echo "Renaming completed"
echo "=============================================="