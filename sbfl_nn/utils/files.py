import os
import re
import glob


def ensure_output_directory(directory_path):
    """
    Ensures the specified directory exists and is writable.
    Creates it if it doesn't exist.
    """
    try:
        # Check if directory exists, create if it doesn't
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")

        # Test if the directory is writable
        test_file_path = os.path.join(directory_path, '.permission_test')
        try:
            with open(test_file_path, 'w') as f:
                f.write('test')
            os.remove(test_file_path)
            return True
        except IOError as e:
            raise PermissionError(
                f"Directory is not writable: {directory_path}") from e

    except PermissionError as e:
        raise PermissionError(f"Permission error: {e}") from e
    except Exception as e:
        raise Exception(f"An error occurred: {e}") from e


def get_pth_files(directory: str) -> list[str]:
    """
    Get all .pth files in the specified directory.
    """
    return glob.glob(os.path.join(directory, "*.pth"))


def get_pkl_files(directory: str) -> list[str]:
    """
    Get all .pkl files in the specified directory.
    """
    return glob.glob(os.path.join(directory, "*.pkl"))


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
