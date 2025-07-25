import logging
import contextlib
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

@contextlib.contextmanager
def log_to_file(message: str, file_path: str):
    """Log a message to a file."""
    with open(file_path, 'a') as f:
        f.write(f"{message}\n")