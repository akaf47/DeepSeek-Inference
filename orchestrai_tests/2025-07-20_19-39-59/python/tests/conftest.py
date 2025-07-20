```python
import pytest
import sys
import os
from unittest.mock import Mock, patch
import tempfile
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def test_config():
    """Session-wide test configuration"""
    return {
        "model_name": "deepseek-coder",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": "test_api_key_12345",
        "base_url": "https://api.deepseek.com/v1",
        "timeout": 30
    }


@pytest.fixture(scope="session")
def mock_api_response():
    """Session-wide mock API response"""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek-coder",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 50,
            "total_tokens": 70
        }
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_requests():
    """Mock requests module for API calls"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Mocked response"}}]
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def sample_prompts():
    """Collection of sample prompts for testing"""
    return [
        "Write a Python function to calculate factorial",
        "Explain how binary search works",
        "Debug this code: print('hello world'",
        "Optimize this sorting algorithm",
        "Create a class for managing user data",
        "Write unit tests for a calculator function"
    ]


@pytest.fixture
def invalid_configs():
    """Collection of invalid configurations for testing"""
    return [
        {},  # Empty config
        {"model_name": "test"},  # Missing api_key
        {"api_key": "test"},  # Missing model_name
        {"model_name": "", "api_key": "test"},  # Empty model_name
        {"model_name": "test", "api_key": ""},  # Empty api_key
        {"model_name": "test", "api_key": "test", "temperature": 2.0},  # Invalid temperature
        {"model_name": "test", "api_key": "test", "max_tokens": -1},  # Invalid max_tokens
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['DEEPSEEK_API_KEY'] = 'test_key_for_testing'
    
    yield
    
    # Cleanup after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']
    if 'DEEPSEEK_API_KEY' in os.environ:
        del os.environ['DEEPSEEK_API_KEY']


@pytest.fixture
def capture_output():
    """Capture stdout and stderr for testing"""
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    yield stdout_capture, stderr_capture
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr
```