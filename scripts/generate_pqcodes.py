import os
import time
import numpy as np
from spiq.streamer.data_streamer import DataStreamer
from spiq.utils.fingerprints import  FingerprintCalculator
from spiq.utils.helper_functions import format_time, save_chunk
from spiq.encoder.encoder import PQEncoder
import time
import gc
import logging
logger = logging.getLogger(__name__)
import argparse
import joblib
import glob
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process with flexible options.")
    parser.add_argument('--fp-path', type=str, required = True,  help="The path to the directory containing the fingerprint files")
    parser.add_argument('--smiles-path', type=str, required = True,  help="The path to a file or directory containing SMILES files.") 
    parser.add_argument('--output-path', type=str, default = '.', help="Output path for the fingerprints, pq-codes and other data generated")
    parser.add_argument('--chunksize', type=int, default=1_000_000, help=" The number of lines to include in each yielded chunk. Defualt 100k ")
    parser.add_argument('--col-idx', type=int, default=0, help="Column index for the smiles in the input data. This is for cases where the input data contains multiple columns. Default is 0 ")    
    parser.add_argument('--file-format', type=str, default='parquet', help="File format for the fingerprints output file. Default is parquet")
    parser.add_argument('--pq-model', type=str,required=True,  help="Path to the trained PQEncoder data.") 
    parser.add_argument('--verbose', type=int, default=1, help="Level of verbosity. Default is 1")
    parser.add_argument('--debug', type=bool, default=False, help="For running the transformatino with all the assert methods. Introduces safety checks and better error output at the expense of time. Default is False")
    return parser.parse_args()


def transform(fp_path, smiles_path, chunksize, output_path, verbose, col_idx, file_format):
    """
    Method to transform every fingerprint into PQ Codes.
    It also saves the PQ-code with the original SMILES code to 
    be able to map PQ-code back to the original molecule. 

    I could have made the decision to save the smiles along with the fp
    in the `calculate_fingerprints` and not have to read them again
    but I think is cleaner to save the fp alone instead of saving them with the SMILES
    since it would be redundant:
    we can go from fp -> smiles and smiles -> fp, but we can't go from 
    pqcodes -> smiles so it makes for me more sense to save smiles+pqcodes
    rather than smiles+fp 
    Another reason is that if I did smiles+fp then only .parquet is suitable since
    .npy only works with binary data.
    """
    ds = DataStreamer()
    fp_calc = FingerprintCalculator()
    # We replicate the process that we did in the fingerprint calculation so we are sure
    # we are reading the smiles in the same order than the fingerprint chunks  
    # if you are loading the fingeprints make sure you read them in the same order as you read the smiles 
    pq_code_output_path = os.path.join(output_path, 'pq_codes')
    N = int(9556903082)
    all_pq_codes = np.zeros((N, 8), dtype='uint8') # all the pq codes together, we will use this to later cluster later on
    N_total = 0
    for i, smiles_chunk in enumerate(ds.parse_input(smiles_path, chunksize, verbose=verbose, col_idx=col_idx)):
        print(f"\rTransforming data: {i} chunks transformed.", end='', flush=True)
        smiles_pq_code_dataframe = pd.DataFrame({
            "smiles": smiles_chunk
        })

        smiles_chunk_fp = pd.read_parquet(os.path.join(fp_path, f'fingerprints_chunk_{i:04d}.{file_format}'))

        pq_code = pq_encoder.transform(smiles_chunk_fp.to_numpy(), verbose=verbose)
        for pq_code_idx in range(pq_code.shape[1]):
            smiles_pq_code_dataframe[f"pq_code_{pq_code_idx}"] = pq_code[:, pq_code_idx]

        if debug:
            assert len(smiles_chunk_fp) == chunksize, f"The chunksize should be the same as the number of fingerprints in each fingerprint file read. Got chunksize={chunksize:,} for fingerprint files of size: {len(smiles_chunk_fp):,}"
            # Assertion to make sure the smiles chunk read is the one that goes with the SMILES chunk read

            """
            Essentially we pick 3 SMILES (begin, mid, end of file) and check that the fingerprint from that SMILES matches the Fingerprint loaded in the fingerprint chunk
            This introduces an additional 4-5 second delay per iteration for chunks of 1M smiles/fingerprints so it should only be run in low number of iterations or DEBUG mode 
            """
            assert (fp_calc.FingerprintFromSmiles(smiles_pq_code_dataframe['smiles'][0], fp='morgan', fpSize=1024, radius=3) == smiles_chunk_fp.iloc[0].to_numpy().reshape(1, -1)).all(), "Reading of SMILES was not consistent with reading of fingerprint chunks"
            assert (fp_calc.FingerprintFromSmiles(smiles_pq_code_dataframe['smiles'][round(chunksize/2)], fp='morgan', fpSize=1024, radius=3) == smiles_chunk_fp.iloc[round(chunksize/2)].to_numpy().reshape(1, -1)).all(), "Reading of SMILES was not consistent with reading of fingerprint chunks"
            assert (fp_calc.FingerprintFromSmiles(smiles_pq_code_dataframe['smiles'][(chunksize-1)], fp='morgan', fpSize=1024, radius=3) == smiles_chunk_fp.iloc[chunksize-1].to_numpy().reshape(1, -1)).all(), "Reading of SMILES was not consistent with reading of fingerprint chunks"

            """
            Another assertion. Similar to the first one. Make sure that going from smiles -> fingerprint returns the same PQ-code as going from
            loaded fp -> PQ-code
            """
            assert (smiles_pq_code_dataframe.iloc[0].to_numpy()[1:] == pq_encoder.transform(fp_calc.FingerprintFromSmiles(smiles_pq_code_dataframe['smiles'][0], fp='morgan', fpSize=1024, radius=3), verbose=0)).all(), "PQ code generated from loaded fingerprint, don't match the PQ code generated from the SMILES in the dataframe"
            assert (smiles_pq_code_dataframe.iloc[int(chunksize/2)].to_numpy()[1:] == pq_encoder.transform(fp_calc.FingerprintFromSmiles(smiles_pq_code_dataframe['smiles'][int(chunksize/2)], fp='morgan', fpSize=1024, radius=3), verbose=0)).all(), "PQ code generated from loaded fingerprint, don't match the PQ code generated from the SMILES in the dataframe"
            assert (smiles_pq_code_dataframe.iloc[int(chunksize-1)].to_numpy()[1:] == pq_encoder.transform(fp_calc.FingerprintFromSmiles(smiles_pq_code_dataframe['smiles'][int(chunksize-1)], fp='morgan', fpSize=1024, radius=3),verbose=0)).all(), "PQ code generated from loaded fingerprint, don't match the PQ code generated from the SMILES in the dataframe"
        
        """
        final dataframe will be of len == chunksize with `m` columns
        and every pq_code is in R^`k` 

        smiles | pq_code_0 | pq_code_1 | ... | pq_code_m |
        --------------------------------------------------
        CCCO   |  232       | 10       | 24  | 144       |         
        """

        all_pq_codes[(i*chunksize):((i+1)*chunksize), :] = pq_code
        # Assert that chunk was appended successfully by checking is not all zeros 
        assert (all_pq_codes[((i+1)*chunksize)-1, :] != np.zeros(pq_code[0].shape)).all
        
        smiles_pq_code_dataframe_output_path = os.path.join(pq_code_output_path, f'pq_code_chunk_{i:05d}.parquet')
        os.makedirs(pq_code_output_path, exist_ok=True)
        smiles_pq_code_dataframe.to_parquet(smiles_pq_code_dataframe_output_path)
        del smiles_chunk, smiles_chunk_fp, smiles_pq_code_dataframe
        gc.collect()

    try: 
        st = time.perf_counter()
        np.save(os.path.join(output_path, 'all_pq_codes'), all_pq_codes)
        end = time.perf_counter()
        print("np save took: ", (end -st))
    except Exception as e:
        print(f"AN exception ocurred while saving with np.save", e)

    try: 
        st = time.perf_counter()
        joblib.dump(all_pq_codes, os.path.join(output_path, 'all_pq_codes.joblib'))
        end= time.perf_counter()
        print("Joblib save took: ", (end - st))
    except Exception as e:
        print("And exception ocurred while saving with joblib: ", e)
        

def main():
    args = parse_arguments()

    global pq_encoder, debug
    debug = args.debug

    try: 
        pq_encoder = joblib.load(args.pq_model)
    except Exception as e:
        raise ValueError('Could not load the pq-model raised exception ', e) 
    s = time.time()

    transform(fp_path=args.fp_path,
              smiles_path=args.smiles_path, 
              chunksize = args.chunksize, 
              output_path=args.output_path, 
              verbose=args.verbose,
              col_idx=args.col_idx, 
              file_format=args.file_format)

    e = time.time()

    print("Generating pqcodes took: ", format_time(e-s))
    
if __name__ == "__main__":
    main()


