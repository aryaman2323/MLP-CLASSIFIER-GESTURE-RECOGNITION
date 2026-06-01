import os
import sys
import shutil
import zipfile
import glob

DATA_DIR = './data'

# Old label mapping used in the original codebase
OLD_LABELS_DICT = {0: 'A', 1: 'B', 2: 'L', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

def migrate_existing_data():
    """Migrate old numbered directories (0, 1, 2...) to uppercase letter directories (A, B, L...)"""
    print("\n[1/3] Checking for existing custom data to migrate...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Created data/ directory.")
        return

    migrated_count = 0
    for idx, letter in OLD_LABELS_DICT.items():
        old_path = os.path.join(DATA_DIR, str(idx))
        new_path = os.path.join(DATA_DIR, letter)

        if os.path.isdir(old_path):
            if os.path.exists(new_path):
                # Merge old path images into new path
                print(f"Merging '{old_path}' into '{new_path}'...")
                for file_name in os.listdir(old_path):
                    src_file = os.path.join(old_path, file_name)
                    dest_file = os.path.join(new_path, f"custom_{file_name}")
                    if os.path.isfile(src_file) and not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
                shutil.rmtree(old_path)
            else:
                # Direct rename
                print(f"Renaming '{old_path}' to '{new_path}'...")
                os.rename(old_path, new_path)
            migrated_count += 1

    if migrated_count > 0:
        print(f"Successfully migrated {migrated_count} custom directories.")
    else:
        print("No old numbered directories to migrate.")

def find_dataset_zip():
    """Look for any zip files in the current folder or downloads folder containing 'asl' or default Kaggle 'archive.zip'"""
    search_paths = [
        './archive.zip',
        './asl-dataset.zip',
        './asl_dataset.zip',
        '../asl-dataset.zip',
        '../asl_dataset.zip',
        os.path.expanduser('~/Downloads/asl-dataset.zip'),
        os.path.expanduser('~/Downloads/asl_dataset.zip')
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    # Check current directory for any zip containing 'asl' or 'archive'
    current_zips = glob.glob("./*asl*.zip") + glob.glob("./*archive*.zip")
    if current_zips:
        return current_zips[0]
        
    return None

def download_via_kaggle():
    """Attempt to download dataset using Kaggle API if credentials exist"""
    print("Checking if Kaggle API can be used to download...")
    try:
        import kaggle
        print("Kaggle package installed, authenticating...")
        kaggle.api.authenticate()
        print("Successfully authenticated with Kaggle. Downloading ayuraj/asl-dataset...")
        kaggle.api.dataset_download_files('ayuraj/asl-dataset', path='.', unzip=False)
        print("Download complete.")
        return './asl-dataset.zip'
    except Exception as e:
        print(f"Could not use Kaggle API automatically: {e}")
        print("Please download the dataset manually.")
        return None

def extract_and_merge_zip(zip_path):
    """Extract zip and merge the folders into ./data/"""
    print(f"\n[2/3] Extracting dataset from '{zip_path}'...")
    
    temp_extract_dir = './temp_asl_extract'
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        print("Extraction complete.")

        print("\n[3/3] Merging extracted dataset into './data/'...")
        # Find directories inside extracted folders.
        # Typically Kaggle datasets are organized under a top level folder like 'asl_dataset' or 'asl-dataset'.
        # Let's search recursively for folders containing ASL classes.
        found_folders = []
        for root, dirs, files in os.walk(temp_extract_dir):
            # If the directory contains image files, let's inspect the directory name
            # Character class directories should be single alphanumeric characters (0-9, a-z)
            # or subdirectories representing classes.
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) > 0:
                dir_name = os.path.basename(root)
                # Check if it looks like a valid class name (length 1, alphanumeric)
                if len(dir_name) == 1 and dir_name.isalnum():
                    found_folders.append((root, dir_name.upper()))

        if not found_folders:
            # Let's try searching for any folder matching a-z, 0-9 in the top levels
            for root, dirs, files in os.walk(temp_extract_dir):
                for d in dirs:
                    if len(d) == 1 and d.isalnum():
                        found_folders.append((os.path.join(root, d), d.upper()))

        if not found_folders:
            print("Error: Could not find any single-character class folders (0-9, A-Z) in the extracted zip.")
            return False

        merged_classes = set()
        total_images_copied = 0
        
        for src_dir, class_name in found_folders:
            dest_dir = os.path.join(DATA_DIR, class_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Copy images
            copied = 0
            for file_name in os.listdir(src_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(src_dir, file_name)
                    dest_file = os.path.join(dest_dir, f"kaggle_{file_name}")
                    # Skip if already exists
                    if not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file)
                        copied += 1
            
            if copied > 0:
                merged_classes.add(class_name)
                total_images_copied += copied
                print(f"  Merged class '{class_name}': added {copied} images from Kaggle.")

        print(f"\nSuccessfully merged {total_images_copied} images across {len(merged_classes)} classes into './data/'!")
        return True

    finally:
        # Clean up temp folder
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)

def main():
    print("=" * 60)
    print(" ASL DATASET IMPORT & MIGRATION UTILITY")
    print("=" * 60)

    # 1. Migrate old numbering structure to letters
    migrate_existing_data()

    # 2. Look for zip
    zip_path = find_dataset_zip()
    
    if zip_path:
        print(f"Found local dataset ZIP file: {zip_path}")
    else:
        print("No local dataset ZIP file found in current folder or Downloads.")
        # Attempt Kaggle API download
        zip_path = download_via_kaggle()

    if not zip_path:
        print("\n" + "!" * 60)
        print(" Kaggle ASL Dataset Zip Not Found.")
        print(" Please follow these instructions:")
        print(" 1. Download the dataset zip manually from:")
        print("    https://www.kaggle.com/datasets/ayuraj/asl-dataset")
        print(" 2. Place the downloaded 'asl-dataset.zip' file into this directory:")
        print(f"    {os.path.abspath('.')}")
        print(" 3. Re-run this script to automatically extract and merge it.")
        print("!" * 60 + "\n")
        sys.exit(1)

    # 3. Extract and merge zip
    success = extract_and_merge_zip(zip_path)
    if success:
        print("\nASL Dataset successfully integrated into current repository.")
        print("You can now run 'create_dataset.py' to process landmarks and regenerate data.pickle.")
    else:
        print("\nFailed to extract and merge ASL Dataset. Check file integrity.")

if __name__ == '__main__':
    main()
