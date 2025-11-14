import os
import shutil
from pathlib import Path

# --- Configuration ---
# 1. Set the two folder paths. Use raw strings (r"...") for Windows paths.
SOURCE_FOLDER_PATH = r"C:\Users\DiegoPC\Documents\project\FruitClassification\data\fruits-360_100x100\fruits-360\Test"
TARGET_FOLDER_PATH = r"C:\Users\DiegoPC\Documents\project\FruitClassification\data\fruits-360_100x100\fruits-360\Training"

# 2. Set to False to perform actual deletions.
#    When True, it will only print what it *would* delete.
DRY_RUN = False
# ---------------------

def get_relative_subdirs(root_dir):
    """
    Walks a directory and returns a set of all relative
    paths to its subdirectories.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"Error: Path is not a directory: {root_path}")
        return set()
        
    subdirs = set()
    for item in root_path.rglob('*'):
        if item.is_dir():
            # Get the path relative to the root directory
            relative_path = item.relative_to(root_path)
            # Store as a platform-independent string
            subdirs.add(str(relative_path))
            
    return subdirs

def sync_deletions(source_dir, target_dir, dry_run=True):
    """
    Deletes directories from target_dir that do not exist in source_dir.
    """
    print("--- Starting Directory Sync ---")
    print(f"Source (Model): {source_dir}")
    print(f"Target (To Clean): {target_dir}")
    
    if dry_run:
        print("\n*** DRY RUN mode is ON. No files or folders will be deleted. ***\n")
    else:
        print("\n*** WARNING: DRY RUN is OFF. Deletions will be permanent. ***\n")

    # Get sets of relative directory paths
    try:
        source_dirs = get_relative_subdirs(source_dir)
        target_dirs = get_relative_subdirs(target_dir)
    except Exception as e:
        print(f"An error occurred while scanning directories: {e}")
        return

    if not source_dirs and Path(source_dir).is_dir():
        print(f"Note: Source directory '{source_dir}' contains no subfolders.")
    
    if not target_dirs and Path(target_dir).is_dir():
        print(f"Note: Target directory '{target_dir}' contains no subfolders. Nothing to do.")
        print("--- Sync Complete ---")
        return

    # Find directories that are in Target but NOT in Source
    dirs_to_delete = target_dirs - source_dirs
    
    if not dirs_to_delete:
        print("All subdirectories match. Nothing to delete.")
        print("--- Sync Complete ---")
        return

    print(f"Found {len(dirs_to_delete)} subdirectories to remove from Target:")

    # Sort by path length (deepest first) to delete nested folders first
    # This prevents errors from trying to delete a non-empty parent dir
    sorted_dirs_to_delete = sorted(list(dirs_to_delete), key=len, reverse=True)

    deleted_count = 0
    error_count = 0

    for rel_path_str in sorted_dirs_to_delete:
        full_path_to_delete = Path(target_dir) / rel_path_str
        
        if dry_run:
            print(f"[DRY RUN] Would delete: {full_path_to_delete}")
            deleted_count += 1
        else:
            try:
                # Use shutil.rmtree to delete a directory and all its contents
                shutil.rmtree(full_path_to_delete)
                print(f"DELETED: {full_path_to_delete}")
                deleted_count += 1
            except OSError as e:
                # OSError is common (e.g., file in use, permissions)
                print(f"ERROR: Could not delete {full_path_to_delete}. Reason: {e}")
                error_count += 1
            except Exception as e:
                print(f"UNEXPECTED ERROR: Could not delete {full_path_to_delete}. Reason: {e}")
                error_count += 1
                
    print("\n--- Sync Summary ---")
    if dry_run:
        print(f"[DRY RUN] Would have deleted {deleted_count} directories.")
    else:
        print(f"Successfully deleted {deleted_count} directories.")
        if error_count > 0:
            print(f"Failed to delete {error_count} directories. See errors above.")
    print("--- Sync Complete ---")

if __name__ == "__main__":
    # Ensure paths are valid before starting
    if not Path(SOURCE_FOLDER_PATH).is_dir() or not Path(TARGET_FOLDER_PATH).is_dir():
        print(f"Error: One or both paths are invalid. Please check your configuration.")
        print(f"Source: {SOURCE_FOLDER_PATH} (Exists: {Path(SOURCE_FOLDER_PATH).is_dir()})")
        print(f"Target: {TARGET_FOLDER_PATH} (Exists: {Path(TARGET_FOLDER_PATH).is_dir()})")
    else:
        sync_deletions(SOURCE_FOLDER_PATH, TARGET_FOLDER_PATH, dry_run=DRY_RUN)