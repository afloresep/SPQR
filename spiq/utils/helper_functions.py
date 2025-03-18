import logging
import os

def _process_input(input_paths):
    """
    Method to deal with input paths for files and folders. 
    Yields files if tuple or list of paths are provided
    input_paths : str, list, or tuple
        A single file path, a folder path, or a list of file and/or folder paths.
        Folders will be traversed to yield file paths contained within.    Yields:
    -------
    str
        File paths from the provided input(s).
    """
    if isinstance(input_paths, str):
        # Single path (file or folder)
        input_paths = [input_paths]
    elif not isinstance(input_paths, (list, tuple)):
        logging.error(f"Invalid input type: {type(input_paths)}. Provide a string, list, or tuple. Skipping")

    for input_path in input_paths:
        if os.path.isdir(input_path):
            # Process all files in the directory
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path):  # Optional: filter by file extensions
                    yield file_path
        elif os.path.isfile(input_path):
            # Process a single file
            yield input_path
        else:
            raise ValueError(f"Invalid input path: {input_path}. Ensure the path exists and is a valid file or directory.")



def format_time(seconds):
    """Simpel script to transform seconds into readable format

    Args:
        seconds (int): Seconds 

    Returns:
        str: (int) h, (int) min, (float) seconds 
    """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours)} h; {int(minutes)}min; {seconds:.2f} s" 