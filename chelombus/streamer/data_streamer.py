import os
from typing import Iterator, List, Optional

from chelombus.utils.helper_functions import _process_input

class DataStreamer:
    """
    Class for streaming large datasets in manageable chunks.
    """
    def parse_input(self, input_path: str, chunksize: Optional[int] = None, verbose: int = 0,
                    smiles_col: int = 0, id_col: Optional[int] = None,
                    with_ids: bool = False) -> Iterator[List]:
        """
        Read input data from a file or a directory of files and yield it in chunks.

        The input path can be a single file or a directory of files. Plain-text
        files are read line by line; the whitespace-separated column at
        `smiles_col` (0-based) is taken as the SMILES. So a file with ONE SMILES
        per line works as-is (smiles_col=0), and a file like "id<TAB>smiles"
        works with smiles_col=1. SDF/SD files are streamed with RDKit and turned
        into SMILES on the fly. Items are buffered until `chunksize` is reached,
        then the buffer is yielded; any remainder is yielded at the end of each
        file.

        Ids (optional):
            By default each chunk is a list of SMILES strings (unchanged
            behaviour). If `with_ids=True` (or you pass `id_col`), each chunk is
            instead a list of (id, smiles) tuples, so a label travels WITH every
            molecule:
              * id_col=k    -> the id is column k of the line, kept exactly as-is
                               (a number, a long hash, a code like "Z1234-5678").
              * id_col=None -> the id is the molecule's running number over the
                               whole input - a stable made-up id.
            Ids are always returned as strings.

        Args:
            input_path (str): Path to a file or directory of input files.
            chunksize (Optional[int]): Items per yielded chunk. None -> one chunk
                per file.
            verbose (int): 0 silent, 1 prints each file name.
            smiles_col (int): 0-based column holding the SMILES (text inputs).
                Ignored for SDF, where SMILES come from the molecule blocks.
            id_col (Optional[int]): 0-based column holding the id, or None to make
                one up. Passing it turns on (id, smiles) output.
            with_ids (bool): Yield (id, smiles) tuples instead of bare SMILES.

        Yields:
            List: a chunk of SMILES strings, or of (id, smiles) tuples when ids
            are requested.
        """
        # Sanity check
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Data source {input_path} not found.")

        want_ids = with_ids or id_col is not None
        buffer: List = []
        next_id = 0  # running counter for made-up ids (all files, in order)

        for file_path in _process_input(input_paths=input_path):
            if verbose == 1:
                print("Processing file:", file_path)

            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in {".sdf", ".sd"}:
                records = self._stream_sdf(file_path)
            else:
                records = self._stream_text(file_path, smiles_col, id_col)

            for raw_id, smiles in records:
                if want_ids:
                    # use the file's id if it has one, else the running number
                    buffer.append((raw_id if raw_id is not None else str(next_id), smiles))
                else:
                    buffer.append(smiles)
                next_id += 1
                if chunksize is not None and len(buffer) == int(chunksize):
                    yield buffer[:]
                    buffer.clear()

            # Yield whatever is left from this file.
            if buffer:
                yield buffer[:]
                buffer.clear()

    def _stream_text(self, file_path: str, smiles_col: int,
                     id_col: Optional[int] = None) -> Iterator[tuple]:
        """Yield (raw_id, smiles) per line. raw_id is the value in column
        `id_col`, or None when there is no id column. Blank / too-short lines
        (no column at `smiles_col`) are skipped."""
        with open(file_path, "r") as file:
            for line in file:
                parts = line.split()
                if smiles_col >= len(parts):
                    continue
                smiles = parts[smiles_col]
                raw_id = parts[id_col] if (id_col is not None and id_col < len(parts)) else None
                yield raw_id, smiles

    def _stream_sdf(self, file_path: str) -> Iterator[tuple]:
        try:
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(
                "RDKit is required to read SDF files. Please install rdkit to stream SDF inputs."
            ) from exc

        # ForwardSDMolSupplier streams molecules without loading the full file in memory
        with open(file_path, "rb") as file_handle:
            supplier = Chem.ForwardSDMolSupplier(file_handle)
            for mol in supplier:
                if mol is None:
                    continue
                try:
                    smiles = Chem.MolToSmiles(mol)
                except Exception as e:
                    print(f"\nFailed to convert molecule in {file_path} to SMILES: {e}\n")
                    continue
                # Use the SD record title as the id when present.
                name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
                yield (name or None), smiles
