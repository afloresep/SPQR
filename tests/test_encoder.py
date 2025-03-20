import pytest
import numpy as np

from spiq.encoder.encoder import PQEncoder 

def test_fit_successful():
    """
    Test that the fit method runs successfully with proper input data
    and that the codebook is set with the expected shape and the 
    encoder_is_trained flag becomes True.
    """
    np.random.seed(42)
    N, D = 100, 16
    X_train = np.random.rand(N, D).astype(np.float32)
    k = 4
    m = 4
    
    encoder = PQEncoder(k=k, m=m, iterations=10)
    encoder.fit(X_train)
    
    assert encoder.encoder_is_trained is True, "Encoder should be trained after successful fit."
    assert encoder.codebook.shape == (m, k, D // m), (
        f"Expected codebook shape to be {(m, k, D//m)}, "
        f"got {encoder.codebook.shape}"
    )
    # Ensure dtype is float per the code
    assert encoder.codebook.dtype == float, "Codebook dtype should be float."


def test_fit_raises_assertionerror_for_non2d_input():
    """
    Test that providing a 1D input array to fit raises an AssertionError.
    """
    X_train_1d = np.array([1, 2, 3])  # 1D input
    encoder = PQEncoder(k=4, m=2, iterations=10)
    
    with pytest.raises(AssertionError, match="The input can only be a matrix"):
        encoder.fit(X_train_1d)


def test_fit_raises_assertionerror_for_3d_input():
    """
    Test that providing a 3D input array to fit raises an AssertionError.
    """
    X_train_3d = np.random.rand(10, 4, 4)
    encoder = PQEncoder(k=4, m=2, iterations=10)
    
    with pytest.raises(AssertionError, match="The input can only be a matrix"):
        encoder.fit(X_train_3d)


def test_fit_raises_assertionerror_for_k_too_large():
    """
    Test that if k >= N, an assertion error is raised.
    """
    N, D = 8, 16
    X_train = np.random.rand(N, D).astype(np.float32)
    
    # Here k == 8, so the assertion says it must be strictly less (k < N).
    encoder = PQEncoder(k=N, m=4, iterations=10)
    
    with pytest.raises(AssertionError, match="should be more than the number of centroids"):
        encoder.fit(X_train)


def test_fit_raises_assertionerror_for_incorrect_dim_division():
    """
    Test that if D is not divisible by m, an assertion error is raised.
    """
    N, D = 100, 10
    X_train = np.random.rand(N, D).astype(np.float32)
    # Here m=3, but D=10 which is not divisible by 3
    encoder = PQEncoder(k=4, m=3, iterations=10)
    
    with pytest.raises(AssertionError, match="dimension should be divisible by the number of subvectors"):
        encoder.fit(X_train)


def test_fit_raises_assertionerror_for_multiple_fits():
    """
    Test that calling fit on an already fitted encoder raises an assertion error.
    """
    N, D = 100, 16
    X_train = np.random.rand(N, D).astype(np.float32)
    encoder = PQEncoder(k=4, m=4, iterations=10)
    
    encoder.fit(X_train)  # First fit
    assert encoder.encoder_is_trained is True
    
    with pytest.raises(AssertionError, match="Encoder can only be fitted once"):
        encoder.fit(X_train)  # Second fit should raise error