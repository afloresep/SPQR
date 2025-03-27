# Little script to try different methods to save the fingeprints
import numpy as np
import time
import pickle
import joblib

# Create a large dummy array (adjust size for testing)
training_data = np.empty([20000000, 1024], dtype='uint8')

def benchmark(method_name, save_func, load_func, path):
    start = time.time()
    save_func(training_data, path)
    save_time = time.time() - start

    start = time.time()
    _ = load_func(path)
    load_time = time.time() - start

    print(f"{method_name:20} | Save: {save_time:.3f}s | Load: {load_time:.3f}s ")

# Method 1: np.save
benchmark("np.save",
          lambda arr, path: np.save(path, arr),
          lambda path: np.load(path + '.npy'),
          "data_npy")

# Method 2: np.savez_compressed
benchmark("np.savez_compressed",
          lambda arr, path: np.savez_compressed(path, arr=arr),
          lambda path: np.load(path + '.npz')['arr'],
          "data_npz")

# Method 3: pickle
benchmark("pickle",
          lambda arr, path: pickle.dump(arr, open(path, 'wb')),
          lambda path: pickle.load(open(path, 'rb')),
          "data_pickle.pkl")

# Method 4: joblib
benchmark("joblib",
          lambda arr, path: joblib.dump(arr, path),
          lambda path: joblib.load(path),
          "data_joblib.pkl")

# Method 5: parquet 
import pandas as pd

def save_parquet(arr, path):
    df = pd.DataFrame(arr)
    df.to_parquet(path, engine='pyarrow')

def load_parquet(path):
    df = pd.read_parquet(path, engine='pyarrow')
    return df.to_numpy()

benchmark("Parquet (pyarrow)",
          save_parquet,
          load_parquet,
          "data_parquet.parquet")