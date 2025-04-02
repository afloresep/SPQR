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
    parser.add_argument('--output-path', type=str, default = '.', help="Output path for the fingerprints, pq-codes and other data generated")
    parser.add_argument('--pq-model', type=str,required=True,  help="Path to the trained PQEncoder data.") 
    parser.add_argument('--verbose', type=int, default=1, help="Level of verbosity. Default is 1")
    parser.add_argument('--debug', type=bool, default=False, help="For running the transformatino with all the assert methods. Introduces safety checks and better error output at the expense of time. Default is False")
    return parser.parse_args()

def transform(fp_path:str, output_path:str,N:int=20_000_000_000):
    
    count = 0
    import glob
    path = os.path.join(fp_path, '*.parquet')
    fp_files = glob.glob(path)
    print(f"Reading {len(fp_files)} files ")
    
    """
    Passing the number of samples (N) is not necessary although it can be passed.
    Instead having to pass the number of pq codes to generate (N) which can be complex/large 
    -thus annoying to remember or type- we just create a sufficiently large array of zeros 
    and start populating as we process each chunk. This process is lazy so we won't run into OOM
    At the end we just passed the actual number of pqcodes processed. 
    """
    import time

    pq_codes = np.zeros((N, pq_encoder.m), dtype=pq_encoder.codebook_dtype)
    for index, fp_file in enumerate(fp_files):
        
        st = time.perf_counter() 
        fp_chunk_df = pd.read_parquet(fp_file)
        fp_chunk = fp_chunk_df.drop(columns='smiles').to_numpy()
        arr_pq_codes = pq_encoder.transform(fp_chunk, verbose=0)
        for idx in range(pq_encoder.m):
            fp_chunk_df[f'pq_code_{idx}'] = arr_pq_codes[:,idx]

        # Random checks to test everything went accordingly 
        # takes 0.1s for range 100. Lower if theres a lot of files
        for i in range(100):
            randint = np.random.randint(0, 1000000)
            rnd_smiles_fp= fp_chunk_df.iloc[randint][1:1025].to_numpy()
            rnd_smiles_pqcode = fp_chunk_df.iloc[randint][1025:1034].to_numpy()
            rnd_smiles_generated_pqcode  = pq_encoder.transform(rnd_smiles_fp.reshape(1, -1), verbose=0)

            assert (rnd_smiles_generated_pqcode == rnd_smiles_pqcode).all()

        fp_chunk_df.to_parquet(os.path.join(output_path, f'pqcode_n_smiles_{index}.parquet'))
        pq_codes[count:(count+arr_pq_codes.shape[0]),:] = arr_pq_codes
        count += len(fp_chunk_df)
        end = time.perf_counter()
        print(f'\rPQ codes generated: {count:,} in {(end - st):.2f} seconds', end='', flush=True) 

    print("\nTransformation completed, saving pq_codes")
    pq_codes = pq_codes[:count]
    np.save('pq_codes_test', pq_codes)

def main():
    args = parse_arguments()

    global pq_encoder, verbose, debug
    verbose = args.verbose
    debug = args.debug
    try: 
        pq_encoder = joblib.load(args.pq_model)
    except Exception as e:
        raise ValueError('Could not load the pq-model. Raised exception ', e) 
    s = time.time()
    transform(fp_path=args.fp_path, 
              output_path=args.output_path)
    e = time.time()

    print("\nGenerating PQ codes took: ", format_time(e-s))
    
if __name__ == "__main__":
    main()


