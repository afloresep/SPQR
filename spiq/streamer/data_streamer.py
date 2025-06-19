import os
from typing import Optional,  Iterator, List
from spiq.utils.helper_functions import _process_input

class DataStreamer:
    """
    Class for streaming large datasets in manageable chunks.
    """
    def parse_input(self, input_path:str, chunksize: Optional[int]=None, verbose:int=0, smiles_col:int=0) -> Iterator[List[str]]: 
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
            verbose (int): Level of verbosity. Default is 0.
            col_idx (int): Column index for the smiles in the input data. This is for cases where the input data contains multiple columns. Default is 0. 

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
            if verbose == 1:
                print(f"Processing file:", file_path)
            with open(file_path, 'r') as file:               
                for line in file:
                    try:
                        buffer.append(line.split()[smiles_col])
                        if chunksize is not None and len(buffer) == int(chunksize):
                            yield buffer[:]
                            buffer.clear()
                    except Exception as e:
                        print(f"\nAn exception occured with line: {line}. Raised Error: {e}")
                        print("\n")
                        continue
        # Process remaining items in the buffer
        # or in case no chunksize was provided yield the whole thing
                if buffer:
                     yield buffer[:]
                     buffer.clear()