import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "microsoft/Phi-3-mini-4k-instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("SUCCESS")