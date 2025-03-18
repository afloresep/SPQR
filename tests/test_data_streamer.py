import os
import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Iterator, List, Optional
from spiq.streamer.data_streamer import DataStreamer


def test_parse_input_with_chunksize(tmp_path: Path):
    # Create a temporary file with known content.
    file_content = ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]
    file_path = tmp_path / "test.txt"
    file_path.write_text("".join(file_content))
    
    streamer = DataStreamer()
    chunksize = 2
    # Expected chunks: [first two lines], [next two lines], [last line]
    result = list(streamer.parse_input(str(file_path), chunksize))
    assert result == [file_content[:2], file_content[2:4], file_content[4:]]
 
def test_parse_input_without_chunksize(tmp_path: Path):
    # Test when chunksize is None: the entire file should be returned as one chunk.
    file_content = ["line1\n", "line2\n", "line3\n"]
    file_path = tmp_path / "test.txt"
    file_path.write_text("".join(file_content))
    
    streamer = DataStreamer()
    result = list(streamer.parse_input(str(file_path), None))
    # Expect a single chunk containing all lines.
    assert result == [file_content]
    
def test_parse_input_file_not_found(tmp_path: Path):
    # Test that a FileNotFoundError is raised when the file does not exist.
    non_existent_path = tmp_path / "nonexistent.txt"
    streamer = DataStreamer()
    with pytest.raises(FileNotFoundError):
        list(streamer.parse_input(str(non_existent_path), 2))