#!/bin/bash

# source
PROJECT_ROOT="/home/s2882278/Diss/lung-tumour"

# .zip files to copy
DATA_ZIP="$PROJECT_ROOT/data.zip"

# target
UNIQUE_ID="$1"
TARGET_DIRECTORY="/disk/scratch/s2882278/lung-tumour/$UNIQUE_ID"
echo "Target directory: $TARGET_DIRECTORY"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIRECTORY"

# Copy .zip files to target directory
cp "$DATA_ZIP" "$TARGET_DIRECTORY"

echo "Files copied to $TARGET_DIRECTORY"

# unzip
cd "$TARGET_DIRECTORY"
unzip -o data.zip

echo "Unzipping done"