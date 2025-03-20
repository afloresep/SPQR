import numpy as np
from .encoder_base import PQEncoderBase
from tqdm import tqdm
from sklearn.cluster import KMeans

class PQEncoder(PQEncoderBase):
    """
    Class to encode high-dimensional vectors into PQ-codes.
    """
    def __init__(self, k:int=256, m:int=8, iterations=20):
        """ Initializes the encoder with trained sub-block centroids.

        Args:
            k (int): Number of centroids. Default is 256. We assume that all subquantizers 
            have the same finit number (k') of reproduction values. So k = (k')^m 
            m (int): Number of distinct subvectors of the input X vector. 
            m is a subvector of dimension D' = D/m where D is the 
            dimension of the input vector X. D is therefore a multiple of m
            iterations (int): Number of iterations for the k-means

        High values of k increase the computational cost of the quantizer as well as memory 
        usage of strogin the centroids (k' x D floating points). Using k=256 and m=8 is often
        a reasonable choice. 

        Reference: DOI: 10.1109/TPAMI.2010.57

        """

        self.m = m 
        self.k = k 
        self.iterations = iterations
        self.encoder_is_trained = False
        self.codebook = None # The codebook is defined as the Cartesian product of all centroids
        self.codebook_dtype =  np.uint8 if self.k <= 2**8 else (np.uint16 if self.k<= 2**16 else np.uint32)

        """ The codebook is defined as the Cartesian product of all centroids. 
        Storing the codebook C explicitly is not efficient, the full codebook would be (k')^m  centroids
        Instead we store the K' centroids of all the subquantizers (k' · m). We can later simulate the full 
        codebook by combining the centroids from each subquantizer. 
        """

        
    def fit(self, X_train:np.array)->None:
        """ KMeans fitting of every subvector matrix from the X_train matrix. Populates 
        the codebook by storing the cluster centers of every subvector

        X_train is the input matrix. For a vector that has dimension D  
        then X_train is a matrix of size (N, D)
        where N is the number of rows (vectors) and D the number of columns (dimension of
        every vector i.e. fingerprint in the case of molecular data)

        Args:
           X_train(np.array): Input matrix to train the encoder.  
        """
        
        assert X_train.ndim == 2, "The input can only be a matrix (X.ndim == 2)"
        N, D = X_train.shape # N number of input vectors, D dimension of the vectors
        assert self.k < N, "the number of training vectors (N for N,D = X_train.shape) should be more than the number of centroids (K)"
        assert D % self.m == 0, f"Vector (fingeprint) dimension should be divisible by the number of subvectors (m). Got {D} / {self.m}" 
        self.D_subvector = int(D / self.m) # Dimension of the subvector. 

        assert self.encoder_is_trained == False, "Encoder can only be fitted once"

        self.codebook = np.zeros((self.m, self.k, self.D_subvector), dtype=float)
        # Divide the input vector into m subvectors 
        for subvector_idx in tqdm(range(self.m), desc='Training PQEncoder'):
            subvector_dim = int(D / self.m) 
            X_train_subvector = X_train[:, subvector_dim * subvector_idx : subvector_dim * (subvector_idx + 1)] 
            # For every subvector, run KMeans and store the centroids in the codebook 
            self.pq_trained= KMeans(n_clusters=self.k, init='k-means++', max_iter=self.iterations).fit(X_train_subvector)

            # Results for training 1M 1024 dimensional fingerprints were: 5 min 58 s, with k=256, m=4, iterations=200, this is much faster than using scipy
            self.codebook[subvector_idx] = self.pq_trained.cluster_centers_

        self.encoder_is_trained = True
       
    def transform(self, X_test):
        """ Transform vectors into PQ code

        Args:
            X (_type_): _description_
        """
        


    def inverse_transform(self, X_test, binary=False):
        """ Inverse transform. From PQ-code to 
        original vector. This process is lossy. 
        If binary=True then the vectors will be returned in binary. 
        This is for the case where our original vectors were binary. 
        If we don't have binary=True then the returned vectors are probably
        like: [0.32134, 0.8232, 0.0132, ... 0.1432, 1.19234] so binary True
        transforms it to 
        [0, 1, 0, ..., 0, 1]

        Args:
            X (_type_): _description_
        """
        
        X_inversed = 0
        reconstructed_binary = (X_test>= 0.5).astype(int)
        return reconstructed_binary 
    