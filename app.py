import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import model_info
import torch
import os

def optimal_loading(model_name, device, offload_folder=None):
    """Smart model loading with auto fallback logic"""
    try:
        # Prefer safetensors if available
        if model_info(model_name).safetensors:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                use_safetensors=True,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32
            )
    except:
        pass

    # Fallback to disk offloading if needed
    os.makedirs(offload_folder, exist_ok=True) if offload_folder else None
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        offload_folder=offload_folder,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32
    )

def main():
    parser = argparse.ArgumentParser(description='DeepSeek LLM Inference')
    parser.add_argument('--prompt', default="Hello, world!", type=str)
    parser.add_argument('--model', default="deepseek-ai/deepseek-llm-7b-base", type=str)
    parser.add_argument('--max_tokens', default=100, type=int)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--offload_folder', default='./offload', type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = optimal_loading(args.model, args.device, args.offload_folder)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()