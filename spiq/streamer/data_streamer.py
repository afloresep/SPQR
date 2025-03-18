import os
import numpy as np
from typing import Optional, Iterable, Iterator, Any, List, Callable
from spiq.utils.helper_functions import _process_input, format_time

class DataStreamer:
    """
    Class for streaming large datasets in manageable chunks.
    """
    def parse_input(self, input_path:str, chunksize: Optional[int]=None) -> Iterator[List[str]]: 
        """
        Reads input data from a file or a directory of files and yields the data in chunks.

        This method processes each file provided by the input path, which can be either a single file 
        or a directory containing multiple files. For each file, it reads the content line by line and 
        accumulates the lines in a buffer. When the number of lines in the buffer reaches the specified 
        chunksize, the buffer is yielded and cleared. If `chunksize` is None or if there are remaining 
        lines that do not complete a full chunk after reading a file, the remaining lines are yielded 
        as a final chunk.

        Args:
            input_path (str): The path to a file or directory containing input files.
            chunksize (Optional[int]): The number of lines to include in each yielded chunk. If None, 
                the entire file content is yielded as a single chunk.

        Yields:
            List[str]: A list of lines from the input file(s), where the list length will be equal to 
            `chunksize` except possibly for the final chunk.
        """ 
        # Sanity check
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Data source {input_path} not found.")
            
        buffer = []
        # helper function to get the file path from the provided input(s)  
        for file_path in _process_input(input_paths=input_path):
            with open(file_path, 'r') as file:               
                for line in file:
                  buffer.append(line)
                  if chunksize is not None and len(buffer) == chunksize:
                    yield buffer[:]
                    buffer.clear()
        # Process remaining items in the buffer
        # or in case no chunksize was provided yield the whole thing
                if buffer:
                     yield buffer[:]
                     buffer.clear()