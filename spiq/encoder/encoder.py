import numpy as np
from .encoder_base import PQEncoderBase
from tqdm import tqdm
import time
from typing import List, Optional
from sklearn.cluster import KMeans

class PQEncoder(PQEncoderBase):
    """
    Class to encode high-dimensional vectors into PQ-codes.
    """
    def __init__(self, k:int=256, m:int=8, iterations=20, subvector_dim:Optional[List[int]]=None):
        """ Initializes the encoder with trained sub-block centroids.

        Args:
            k (int): Number of centroids. Default is 256. We assume that all subquantizers 
            have the same finit number (k') of reproduction values. So k = (k')^m 
            m (int): Number of distinct subvectors of the input X vector. 
            m is a subvector of dimension D' = D/m where D is the 
            dimension of the input vector X. D is therefore a multiple of m
            custom_split (List[int]): A custom split for the input vector. Replaces `m` for a custom 
            split. The List holds the number of values per split in sequential order. For example 
                For an input vector [1,2,3,4,5,6,7,8] if we pass custom_split = [2,4,2]
                then the subvector would be:
                    subvector_a = [1,2]
                    subvector_b = [3,4,5,6]
                    subvector_a = [7,8]
                Where as with m = 4 it would have been 
                [[1,2], [3,4], [5,6], [7,8]] 

            iterations (int): Number of iterations for the k-means

        High values of k increase the computational cost of the quantizer as well as memory 
        usage of strogin the centroids (k' x D floating points). Using k=256 and m=8 is often
        a reasonable choice. 

        Reference: DOI: 10.1109/TPAMI.2010.57
        """

        self.k = k 
        self.iterations = iterations
        self.subvector_dims = subvector_dim 
        self.m = m if subvector_dim is None else len(subvector_dim)
        self.encoder_is_trained = False
        self.codebook_dtype =  np.uint8 if self.k <= 2**8 else (np.uint16 if self.k<= 2**16 else np.uint32)
        self.pq_trained = []

        """ The codebook is defined as the Cartesian product of all centroids. 
        Storing the codebook C explicitly is not efficient, the full codebook would be (k')^m  centroids
        Instead we store the K' centroids of all the subquantizers (k' · m). We can later simulate the full 
        codebook by combining the centroids from each subquantizer. 
        """

        
    def fit(self, X_train:np.array, **kwargs)->None:
        """ KMeans fitting of every subvector matrix from the X_train matrix. Populates 
        the codebook by storing the cluster centers of every subvector

        X_train is the input matrix. For a vector that has dimension D  
        then X_train is a matrix of size (N, D)
        where N is the number of rows (vectors) and D the number of columns (dimension of
        every vector i.e. fingerprint in the case of molecular data)

        Args:
           X_train(np.array): Input matrix to train the encoder.  
           **kwargs: Optional keyword arguments passed to the underlying KMeans `fit()` function.
        """
        
        assert X_train.ndim == 2, "The input can only be a matrix (X.ndim == 2)"
        N, D = X_train.shape # N number of input vectors, D dimension of the vectors
        assert self.k < N, "the number of training vectors (N for N,D = X_train.shape) should be more than the number of centroids (K)"
        if self.subvector_dims is None:
            assert D % self.m == 0, f"Vector (fingeprint) dimension should be divisible by the number of subvectors (m). Got {D} / {self.m}" 
        else:
            assert D == np.sum(self.subvector_dims), f"The sum of the subvector dimensions must be the same as the dimension of the input matrix. Got input matrix of dimension {D} =! {np.sum(self.subvector_dims)}"
        self.og_D = D # We save the original dimensions of the input vector (fingerprint) for later use
        assert self.encoder_is_trained == False, "Encoder can only be fitted once"

        self.codebook_cluster_centers = [] 

        if self.subvector_dims is None:
            # Dimensions for every subvector. We store them in a list so transform is compatible with custom subvector dimensions  
            # Since no custom split was passed then each subvector has the same dimension [D/ self.m] 
            self.subvector_dims = [int(D/ self.m)] * self.m
            
        # self.codebook_cluster_centers = np.zeros((self.m, self.k, self.D_subvector), dtype=float)
        # Divide the input vector into m subvectors 
        start_idx = 0 
        for idx, dim in enumerate(tqdm(self.subvector_dims, desc='Training PQEncoder')):
            end_idx = start_idx + dim
            X_train_subvector = X_train[:, start_idx:end_idx]
            # For every subvector, run KMeans and store the centroids in the codebook 
            kmeans = KMeans(n_clusters=self.k, init='k-means++', max_iter=self.iterations, **kwargs).fit(X_train_subvector)
            self.pq_trained.append(kmeans)

            start_idx = end_idx 
            # Results for training 1M 1024 dimensional fingerprints were: 5 min 58 s, with k=256, m=4, iterations=200, this is much faster than using scipy
            # Store the cluster_centers coordinates in the codebook
            self.codebook_cluster_centers.append(kmeans.cluster_centers_)

        self.encoder_is_trained = True
        del X_train # remove initial training data from memory


    def transform(self, X:np.array, verbose:int=1, **kwargs) -> np.array:
        """
        Transforms the input matrix X into its PQ-codes.

        For each sample in X, the input vector is split into `m` equal-sized subvectors. 
        Each subvector is assigned to the nearest cluster centroid
        and the index of the closest centroid is stored. 

        The result is a compact representation of X, where each sample is encoded as a sequence of centroid indices.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features), 
                            where n_features must be divisible by the number of subvectors `m`.
            **kwargs: Optional keyword arguments passed to the underlying KMeans `predict()` function.

        Returns:
            np.ndarray: PQ codes of shape (n_samples, m), where each element is the index of the nearest centroid 
                        for the corresponding subvector.
        """

        assert self.encoder_is_trained == True, "PQEncoder must be trained before calling transform"

        N, D = X.shape
        # Store the index of the Nearest centroid for each subvector
        pq_codes = np.zeros((N, self.m), dtype=self.codebook_dtype)

        # If our original vector is 1024 and our m (splits) is 8 then each subvector will be of dim= 1024/8 = 128
        iterable = enumerate(self.subvector_dims)
        if verbose > 0: 
            iterable = tqdm(iterable, desc='Generating PQ-codes', total=self.m)

        start_idx = 0
        for subvector_idx, dim in iterable:
            end_idx = start_idx + dim
            X_subvector = X[:, start_idx:end_idx]
 
            # Get the kmeans object trained before on every subvector
            kmeans = self.pq_trained[subvector_idx]
            print(f"Using the Kmeans ({subvector_idx}) trained on data [{start_idx}:{end_idx}]")
            # Run predict to get the centroid id for that specific subvector
            pq_codes[:, subvector_idx] = kmeans.predict(X_subvector) 
            start_idx = end_idx
            
        # Free memory 
        del  X

        # Return pq_codes (labels of the centroids for every subvector from the X_test data)
        return pq_codes

    def inverse_transform(self, X_codes:np.array, binary=False):
        """ Inverse transform. From PQ-code to the original vector. 
        This process is lossy so we don't expect to get the exact same data.
        If binary=True then the vectors will be returned in binary. 
        This is useful for the case where our original vectors were binary. 
        With binary=True then the returned vectors are transformed from 
        [0.32134, 0.8232, 0.0132, ... 0.1432, 1.19234] to 
        [0, 1, 0, ..., 0, 1]

        Args:
            pq_code: (np.array): Input data of PQ codes to be transformed into the
            original vectors.
        """
        
        # Get shape of the input matrix of PQ codes
        N, D = X_codes.shape

        # The dimension of the PQ vectors should be the same 
        # as the number of splits (subvectors) from the original data 

        assert D == (self.m), f"The dimension D of the PQ-codes (N,D) should be the same as the number of the subvectors or splits (m) . Got D = {D} for m = {self.m}"
        assert D == (self.og_D  / self.D_subvector), f"The dimension D of the PQ-codes (N,D) should be the same as the original vector dimension divided the subvector dimension"

        X_inversed = np.empty((N, D*self.D_subvector), dtype=float)
        for subvector_idx in range(self.m):
            X_inversed[:, subvector_idx*self.D_subvector:((subvector_idx+1)*self.D_subvector)] = self.codebook_cluster_centers[subvector_idx][X_codes[:, subvector_idx], :]

        # Free memory
        del X_codes

        if binary:
            reconstructed_binary = (X_inversed>= 0.6).astype('int8')
            return reconstructed_binary 

        return X_inversed 