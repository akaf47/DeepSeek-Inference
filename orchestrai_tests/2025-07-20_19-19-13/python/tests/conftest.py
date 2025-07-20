```python
import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session", autouse=True)
def mock_transformers():
    """Mock transformers library for all tests."""
    with patch('app.transformers') as mock_transformers:
        # Mock AutoModelForCausalLM
        mock_model_class = Mock()
        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model_class.from_pret