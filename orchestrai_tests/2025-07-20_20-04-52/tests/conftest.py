```python
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_model_dir():
    """Create a temporary directory for model files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_torch_model():
    """Mock PyTorch model for testing"""
    model = Mock()
    model.eval.return_value = model
    model.forward.return_value = torch.randn(1, 10, 1000)
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing"""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "decoded text"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing"""
    return [
        "def fibonacci(n):",
        "class Calculator:",
        "import torch\nimport numpy as np",
        "# Write a function to sort a list",
        "SELECT * FROM users WHERE"
    ]


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability"""
    with patch('torch.cuda.is_available') as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA unavailability"""
    with patch('torch.cuda.is_available') as mock:
        mock.return_value = False
        yield mock


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def sample_model_outputs():
    """Sample model outputs for testing"""
    return {
        'logits': torch.randn(1, 10, 1000),
        'hidden_states': torch.randn(1, 10, 768),
        'attentions': torch.randn(1, 12, 10, 10)
    }


@pytest.fixture
def performance_config():
    """Configuration optimized for performance testing"""
    return {
        'model_name': 'deepseek-coder-1.3b',
        'max_length': 256,
        'temperature': 0.7,
        'top_p': 0.95,
        'batch_size': 4,
        'use_cache': True
    }
```