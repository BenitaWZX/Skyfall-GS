#!/usr/bin/env python3
"""
COLMAP reconstruction with Z-up fix - GPU DISABLED by default
"""

import os
import logging
from argparse import ArgumentParser
import shutil

parser = ArgumentParser("Colmap converter with Z-up fix")
parser.add_argument("--use_gpu", action='store_true', help="Enable GPU acceleration (default: CPU only)")
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--auto_z_up", action='store_true', 
                    help="Automatically enforce Z-up coordinate system")
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if args.use_gpu else 0  # Default is 0 (CPU only)

def apply_yz_swap_to_colmap_output(sparse_dir):
    """Apply Y-Z swap to fix coordinate system at COLMAP level"""
    print("Applying Y-Z swap to COLMAP reconstruction...")
    
    # Transform points3D.txt
    points_file = os.path.join(sparse_dir, "points3D.txt")
    if os.path.exists(points_file):
        with open(points_file, 'r') as f:
            lines = f.readlines()
        
        transformed = []
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                transformed.append(line)
                continue
            
            parts = line.strip().split()
            if len(parts) >= 4:
                pid = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                rest = ' '.join(parts[4:])
                # Y-Z swap: [X, Y, Z] -> [X, Z, Y]
                new_line = f"{pid} {x} {z} {y} {rest}\n"
                transformed.append(new_line)
            else:
                transformed.append(line)
        
        with open(points_file, 'w') as f:
            f.writelines(transformed)
        print(f"  Transformed points3D.txt")
    
    # Transform images.txt
    images_file = os.path.join(sparse_dir, "images.txt")
    if os.path.exists(images_file):
        with open(images_file, 'r') as f:
            lines = f.readlines()
        
        transformed = []
        in_camera_section = True
        
        for i, line in enumerate(lines):
            if line.startswith('#') or line.strip() == '':
                transformed.append(line)
                if '# Number of images:' in line:
                    in_camera_section = False
                continue
            
            if in_camera_section:
                transformed.append(line)
            else:
                if i % 2 == 0:  # Pose line
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        img_id = parts[0]
                        qw, qx, qy, qz = map(float, parts[1:5])
                        tx, ty, tz = map(float, parts[5:8])
                        camera_id = parts[8]
                        name = parts[9]
                        
                        # Y-Z swap for translation and quaternion
                        new_line = f"{img_id} {qw} {qx} {qz} {qy} {tx} {tz} {ty} {camera_id} {name}\n"
                        transformed.append(new_line)
                    else:
                        transformed.append(line)
                else:
                    transformed.append(line)
        
        with open(images_file, 'w') as f:
            f.writelines(transformed)
        print(f"  Transformed images.txt")
    
    print("Y-Z swap applied to COLMAP reconstruction")

# ===== MAIN RECONSTRUCTION PIPELINE =====
if not args.skip_matching:
    os.makedirs(args.source_path + "/sparse", exist_ok=True)
    
    ## Feature extraction - GPU DISABLED by default
    print(f"Feature extraction (GPU: {'ON' if use_gpu else 'OFF'})...")
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    ## Feature matching - GPU DISABLED by default
    print(f"Feature matching (GPU: {'ON' if use_gpu else 'OFF'})...")
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    ### Bundle adjustment - GPU DISABLED by default
    print(f"Bundle adjustment (GPU: {'ON' if use_gpu else 'OFF'})...")
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    
    # Add GPU flag for bundle adjustment
    if use_gpu:
        mapper_cmd += " --Mapper.ba_global_use_pba 1"
    else:
        mapper_cmd += " --Mapper.ba_global_use_pba 0"
    
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# Copy the input images to the images folder
os.makedirs(args.source_path + "/images", exist_ok=True)
files = os.listdir(args.source_path + "/input")
for file in files:
    source_file = os.path.join(args.source_path, "input", file)
    destination_file = os.path.join(args.source_path, "images", file)
    shutil.copy2(source_file, destination_file)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

# ===== AUTOMATIC Z-UP FIX =====
if args.auto_z_up:
    print("\n=== Applying automatic Z-up correction ===")
    sparse_dir = args.source_path + "/sparse/0"
    
    # First convert to text format
    print("Converting to text format...")
    convert_cmd = (colmap_command + " model_converter \
        --input_path " + sparse_dir + " \
        --output_path " + sparse_dir + " \
        --output_type TXT")
    exit_code = os.system(convert_cmd)
    if exit_code != 0:
        logging.error(f"Model conversion failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    # Apply Y-Z swap to fix coordinate system
    apply_yz_swap_to_colmap_output(sparse_dir)
else:
    # Standard conversion
    print("Converting COLMAP binary files to text format...")
    convert_cmd = (colmap_command + " model_converter \
        --input_path " + args.source_path + "/sparse/0 \
        --output_path " + args.source_path + "/sparse/0 \
        --output_type TXT")
    exit_code = os.system(convert_cmd)
    if exit_code != 0:
        logging.error(f"Model conversion failed with code {exit_code}. Exiting.")
        exit(exit_code)

print("COLMAP files converted to text format.")

if args.resize:
    print("Copying and resizing...")
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    
    files = os.listdir(args.source_path + "/images")
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)
        
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("\n✅ Done! COLMAP reconstruction complete.")
if args.auto_z_up:
    print("   Z-up coordinate system has been enforced.")
    print("   Use this reconstruction directly with Skyfall-GS (no need for Y-Z swap in code).")