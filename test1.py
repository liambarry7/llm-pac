# pip install -U "huggingface_hub"

# hf auth login
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto") # pip install accelerate

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

# https://huggingface.co/google/gemma-2b
