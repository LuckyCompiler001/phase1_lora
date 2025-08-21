from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "NousResearch/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


from datasets import load_dataset
ds = load_dataset("tatsu-lab/alpaca")