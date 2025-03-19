import numpy as np
import pytest
import logging
from functools import partial

# Assuming your module is named "fingerprint_module.py", adjust the import as needed.
from spiq.utils.fingerprints import _calculate_morgan_fp, FingerprintCalculator

# A fixture for common fingerprint parameters.
@pytest.fixture
def fp_params():
    return {'fpSize': 2048, 'radius': 2}


def test_calculate_morgan_fp_valid(fp_params):
    """
    Test that _calculate_morgan_fp returns a NumPy array of the correct shape
    when given a valid SMILES string.
    """
    smiles = "CCO"  
    fingerprint = _calculate_morgan_fp(smiles, **fp_params)
    assert isinstance(fingerprint, np.ndarray)
    assert fingerprint.shape[0] == fp_params['fpSize']


def test_calculate_morgan_fp_invalid(fp_params):
    """
    Test that _calculate_morgan_fp returns a random fingerprint (0s and 1s)
    for an invalid SMILES string.
    """
    smiles = "invalid_smiles"
    fingerprint = _calculate_morgan_fp(smiles, **fp_params)
    
    assert isinstance(fingerprint, np.ndarray)
    assert fingerprint.shape[0] == fp_params['fpSize']
    
    assert np.all(np.isin(fingerprint, [0, 1]))


def test_calculate_morgan_fp_missing_parameters():
    """
    Test that _calculate_morgan_fp raises a ValueError when required parameters are missing.
    """
    smiles = "CCO"
    with pytest.raises(ValueError):
        _calculate_morgan_fp(smiles)  # No fpSize or radius provided.

def test_calculate_morgan_fp_logs_warning(fp_params, caplog):
    """
    Test that a warning is logged when an invalid SMILES string is processed.
    """
    smiles = "invalid_smiles"
    with caplog.at_level(logging.WARNING):
        _calculate_morgan_fp(smiles, **fp_params)
    
    # Check that a warning message was logged.
    assert any("could not be converted" in record.message for record in caplog.records)


def test_fingerprint_calculator_valid(fp_params):
    """
    Test that FingerprintCalculator.FingerprintFromSmiles returns a NumPy array
    with fingerprints for a list of valid SMILES strings.
    """
    calculator = FingerprintCalculator()
    smiles_list = ["CCO", "C1CCCCC1", "O=C=O"]
    fingerprints = calculator.FingerprintFromSmiles(smiles_list, 'morgan', **fp_params)
    
    assert isinstance(fingerprints, np.ndarray)
    assert fingerprints.shape == (len(smiles_list), fp_params['fpSize'])


def test_fingerprint_calculator_invalid_fp(fp_params):
    """
    Test that FingerprintCalculator.FingerprintFromSmiles raises a ValueError
    when an unsupported fingerprint type is requested.
    """
    calculator = FingerprintCalculator()
    smiles_list = ["CCO", "C1CCCCC1"]
    with pytest.raises(ValueError):
        calculator.FingerprintFromSmiles(smiles_list, 'unsupported', **fp_params)