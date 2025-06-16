# #TODO: Create a python implementation of PQKmeans. For now,  we will use the cpp implementation from Yusuke Matsui et al. 
# """reference:
# @inproceedings{pqkmeans,
#     author = {Yusuke Matsui and Keisuke Ogaki and Toshihiko Yamasaki and Kiyoharu Aizawa},
#     title = {PQk-means: Billion-scale Clustering for Product-quantized Codes},
#     booktitle = {ACM International Conference on Multimedia (ACMMM)},
#     year = {2017},
# }"""
# Python implementation of the PQKMeans algorithm

from typing import Type, TypeVar
import numpy as np
from typing import List
import pqkmeans
from spiq import PQEncoder

class PQKMeans:
    def __init__(self, encoder: Type[PQEncoder], k: int, iteration:int=20, verbose:bool= False ):
        """PQKMeans class
        Args:
            encoder (Type[PQEncoder]): trained spiq.encoder.PQEncoder class
            k (int): Number of clusters
            iteration (int, optional): Number of iterations. Defaults to 20.
            verbose (bool, optional): Verbose level. Defaults to False.
        """
        if encoder.encoder_is_trained:
            self.cluster = pqkmeans.clustering.PQKMeans(encoder=encoder,
                                                        k=k, 
                                                        iteration=iteration, 
                                                        verbose=verbose)
        else:
            raise ValueError("Encoder must be trained before clustering")


    def fit(self, X_train):
        self.cluster.fit(X_train)
    
    def predict(self, X):
        return self.cluster.predict(X)

    def fit_predict(self, X)->np.ndarray:
        """Performing clustering on `X` and returns cluster labels.

        Args:
            X (np.array): array-like of shape (n_samples, pq_code_size)

        Returns:
            labels (ndarray): Cluster labels of shape (n_samples,)
        """       
        return self.cluster.fit_predict(X)

# class PQKMeans:
#     def __init__():
#         """
#         - Store codewords (list or array of shape [M, Ks, subdim])
#         - M = number of subspaces
#         - Ks = number of codewords per subspace (often 256)
#         - K = number of clusters
#         - max_iterations = stopping condition (or you can use another criterion)
#         - Possibly compute and store self.distance_matrices_among_codewords (shape [M, Ks, Ks])
#         """
#         self.codewords = codewords
#         self.k = K 
#         self.max_iter = max_iterations
#         self.verbose = verbose
        

#     def _build_distance_matrices(self) -> None:
#         """
#         Precompute the distance between every pair of codewords in each subspace.
#         Store result in self.distance_matrices_among_codewords[m][k1][k2].
#         Each entry is a float for squared distance between those codewords.
#         """
#         pass

#     def _build_pqtable_for_centers(self, centers: np.ndarray) -> SomePQTableStructure:
#         """
#         Build or update an indexing structure (often called a PQTable or inverted multi-index)
#         to quickly assign points to these K centers, especially when K is large (~100,000).
#         Return the data structure that can be used for fast nearest-center lookups.
#         """
#         pass

#     def _assign_points_parallel(
#         self, 
#         pqcodes: np.ndarray,        # shape (N, M), each row is a PQ code
#         table: SomePQTableStructure
#     ) -> np.ndarray:
#         """
#         1. Partition the data among multiple processes.
#         2. Each process:
#             - Loops over its subset of pqcodes
#             - Finds the nearest center (via the PQTable or another structure)
#             - Records an assignment index for each data point
#         3. Collect all partial assignments and combine into a single 1D array of size N.
#         Return that final assignments array.
#         """
#         pass

#     def _update_centers_sparse_voting(
#         self, 
#         pqcodes: np.ndarray,
#         assignments: np.ndarray
#     ) -> np.ndarray:
#         """
#         *Sparse Voting* approach:
#          - For each cluster k:
#              1) Gather all PQ codes assigned to cluster k
#              2) For each subspace m:
#                 - Build a frequency histogram of shape [Ks] (counts of how often each sub-codeword appears)
#                 - Multiply that histogram by the distance matrix for subspace m to find which codeword yields minimal total distance
#                 - That codeword becomes the new center's sub-code index for subspace m
#          - Return the updated cluster centers array (shape [K, M]).
#         """
#         pass

#     def fit(self, pqcodes: np.ndarray) -> None:
#         """
#         Main k-means loop:
#           - pqcodes shape is (N, M), each row is a PQ code in [0..Ks-1].
#           - Initialize centers (randomly pick from pqcodes or another strategy).
#           - For iteration in range(max_iterations):
#               1. Build/update table for the current centers
#               2. Assign points in parallel (using the table)
#               3. Update centers with sparse voting
#               4. (Optional) Check convergence or break if no change
#         Store final assignments and centers internally.
#         """
#         pass

#     def get_assignments(self) -> np.ndarray:
#         """
#         Return the current assignment array (1D, length N).
#         """
#         pass

#     def get_centers(self) -> np.ndarray:
#         """
#         Return the current cluster centers as PQ codes (shape [K, M]).
#         """
#         pass