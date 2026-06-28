import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch
import pytest
from intextus import DenseEncoder

@pytest.fixture
def mock_dependencies():
    temp_dir = tempfile.TemporaryDirectory()
    model_path = os.path.join(temp_dir.name, "model.gguf")
    tokenizer_path = os.path.join(temp_dir.name, "tokenizer.json")
    
    with open(model_path, "wb") as f:
        f.write(b"mock_model_data")
    with open(tokenizer_path, "w") as f:
        f.write('{"model": {"vocab": {}}}')
        
    yield model_path, tokenizer_path
    
    temp_dir.cleanup()

@patch("intextus.encoder.CppIntextusEncoder")
def test_encoder_init_and_encode(mock_cpp_encoder_cls, mock_dependencies):
    model_path, tokenizer_path = mock_dependencies
    
    # Configure mock C++ encoder
    mock_cpp_encoder = MagicMock()
    mock_cpp_encoder_cls.return_value = mock_cpp_encoder
    
    # Mock return values for methods
    dummy_embs = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    
    mock_cpp_encoder.encode.return_value = dummy_embs
    
    # Create encoder (point to the direct gguf file path)
    encoder = DenseEncoder(model_path, tokenizer_path)
    
    # Test encode method
    embs = encoder.encode(["test query 1", "test query 2"], max_length=128, normalize=True)
    mock_cpp_encoder.encode.assert_called_with(["test query 1", "test query 2"], 128, True)
    assert np.array_equal(embs, dummy_embs)

@patch("intextus.encoder.CppIntextusEncoder")
def test_encoder_init_with_directory(mock_cpp_encoder_cls):
    temp_dir = tempfile.TemporaryDirectory()
    gguf_path = os.path.join(temp_dir.name, "model.gguf")
    tokenizer_path = os.path.join(temp_dir.name, "tokenizer.json")
    
    with open(gguf_path, "wb") as f:
        f.write(b"mock_model_data")
    with open(tokenizer_path, "w") as f:
        f.write('{"model": {"vocab": {}}}')
        
    encoder = DenseEncoder(temp_dir.name)
    
    mock_cpp_encoder_cls.assert_called_with(
        gguf_path,
        tokenizer_path,
        False,          # do_lower_case
        0,              # num_threads
        101,            # cls_token_id
        102             # sep_token_id
    )
    
    temp_dir.cleanup()

def test_real_embedding_end_to_end():
    # End-to-end validation with the real default C++ engine and GGUF model
    print("\nRunning real end-to-end embedding test...")
    encoder = DenseEncoder("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test encoding
    texts = ["hello world", "this is a dense integration test"]
    embs = encoder.encode(texts, max_length=128, normalize=True)
    
    # Assert shape: all-MiniLM-L6-v2 has a 384-dimensional embedding space
    assert embs.shape == (2, 384)
    
    # Check L2 normalization (sum of squares is close to 1)
    norm = np.linalg.norm(embs[0])
    assert np.allclose(norm, 1.0, atol=1e-5)
    
    # Check that embeddings are non-zero and distinct
    assert not np.allclose(embs[0], 0.0)
    assert not np.allclose(embs[0], embs[1])

def test_different_quantizations():
    # Verify that we can request and load different quantization types
    print("\nRunning different quantization test...")
    encoder = DenseEncoder("sentence-transformers/all-MiniLM-L6-v2", quantization="Q4_0")
    texts = ["testing quantization loading"]
    embs = encoder.encode(texts)
    assert embs.shape == (1, 384)

