import torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="NousResearch/Llama-2-7b-hf")
ap.add_argument("--adapter", default="models/lora_llama2/adapter")
ap.add_argument("--use_qlora", action="store_true")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, use_fast=False)
if tok.pad_token is None: tok.pad_token = tok.eos_token

load_kwargs = {"device_map": "auto"}
if args.use_qlora:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                            bnb_4bit_compute_dtype=torch.float16,
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_quant_type="nf4")
else:
    load_kwargs["load_in_8bit"] = True

base = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
model = PeftModel.from_pretrained(base, args.adapter)

prompt = "Give me three bullet points about reinforcement learning."
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=120)
print(tok.decode(out[0], skip_special_tokens=True))
