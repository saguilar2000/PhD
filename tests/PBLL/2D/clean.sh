#!/bin/bash

echo "=============================================="
echo "Cleaning directory..."
echo "=============================================="

# Remove Slurm output and error files
echo "Removing *.out and *.e files..."
rm -f *.out *.e

BACKUP_ROOT="./backups"

TARGETS=("condinit_drift_z0" "condinit_drift_z0p5" "condinit_drift_z1" "condinit_nodrift_z0" "condinit_nodrift_z0p5" "condinit_nodrift_z1")

echo "=============================================="
echo "Backing up directories..."
echo "=============================================="

# Ensure the backup folder exists
mkdir -p "$BACKUP_ROOT"

# Calculate the number for the next backup (if none starts with 00)
LATEST=$(ls -d "$BACKUP_ROOT"/backup_* 2>/dev/null | sed 's/.*backup_//' | sort -n | tail -1)
NEXT_NUM=$(( 10#${LATEST:-0} + 1 ))

# Format the name to two digits (e.g., 00, 01, 02)
NEW_BACKUP_DIR="$BACKUP_ROOT/backup_$(printf "%02d" $NEXT_NUM)"

echo "Creating directory: $NEW_BACKUP_DIR"
mkdir -p "$NEW_BACKUP_DIR"

# Move the folders if they exist
for dir in "${TARGETS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Moving '$dir' to '$NEW_BACKUP_DIR'..."
        mv "$dir" "$NEW_BACKUP_DIR/"
    else
        echo "Warning: '$dir' not found. Skipping."
    fi
done

echo "=============================================="
echo "Backup completed in: $NEW_BACKUP_DIR"
echo "=============================================="