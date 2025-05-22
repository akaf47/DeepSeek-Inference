# 🚀 DeepSeek-Inference: Local AI Deployment

Run **DeepSeek LLM models** (7B/67B) locally with GPU/CPU support. Includes ready-to-use inference scripts and API examples.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/OS-Windows%20|%20Linux%20|%20macOS-lightgrey)

---

## Features

- ✅ **Local inference** for `deepseek-llm-7b-base` and other variants
- ✅ **GPU (CUDA) and CPU support**
- ✅ **Simple CLI and API examples**
- ✅ MIT Licensed – Free for commercial use

---

## 🛠️ Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU (Optional, for CUDA acceleration)
- [Git](https://git-scm.com/)

### 1. Clone the Repository
git clone https://github.com/akaf47/DeepSeek-Inference.git
cd DeepSeek-Inference

### 2. Install Dependencies
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt

**Arguments:**
- `--prompt`: Text prompt for the model (default: "Hello, world!")
- `--model`: Model name or path (default: deepseek-ai/deepseek-llm-7b-base)
- `--max_tokens`: Maximum tokens to generate (default: 100)
- `--device`: Device to use: `auto`, `cuda`, or `cpu` (default: auto)
- `--offload_folder`: Folder for model offloading (default: ./offload)


## 🧑‍💻 Example Queries

You can override default values with these example commands:

- **Override the prompt and model:**  
  <pre>
	python app.py --prompt "What is AI?" --model deepseek-ai/deepseek-llm-67b-base
  </pre>

- **Override the maximum number of tokens generated:**  
  <pre>
	python app.py --prompt "Summarize the history of AI." --max_tokens 200
  </pre>

- **Use a specific device (CPU):**  
  <pre>
	python app.py --prompt "List AI applications." --device cpu
  </pre>

- **Specify a custom offload folder:**  
  <pre>
	python app.py --prompt "Explain neural networks." --offload_folder ./my_offload_dir
  </pre>

- **Override all arguments at once:**  
  <pre>
	python app.py --prompt "Describe reinforcement learning." --model deepseek-ai/deepseek-llm-7b-base --max_tokens 150 --device cuda --offload_folder ./custom_offload
  </pre>
  
