import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

import pandas as pd
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate.utils import release_memory
import gc
from transformers import set_seed
import matplotlib.pyplot as plt
from awq import AutoAWQForCausalLM
from utils import save_plot

model_list = ["CodeLlama-34B"]
quants = ["AWQ", "GPTQ"]

def model_memory_footprint(model):
    return float(f"{model.get_memory_footprint() / 1024 / 1024 / 1024:.2f}")


def model_memory_footprint_awq(model):
    mem = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem + mem_bufs
    return float(f"{mem / 1024 / 1024 / 1024:.2f}")


def generate_helper(model, tokenizer, prompt):
    inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to("cuda")

    # warm up
    for i in range(3):
        _ = model.generate(inputs, max_length=15, pad_token_id=tokenizer.eos_token_id)

    start = time.perf_counter()
    generated_tokens = model.generate(
        inputs, max_length=48, pad_token_id=tokenizer.eos_token_id
    )
    duration = time.perf_counter() - start
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    generated_text = generated_text[len(prompt) :]

    latency_per_token_in_ms = (duration / len(generated_tokens[0])) * 1000

    return {
        "text": generated_text,
        "latency": float(f"{round(latency_per_token_in_ms,2)}"),
        "duration": duration,
    }



if __name__ == "__main__":

    prompt = "def factorial("

    response_df = pd.DataFrame(index=model_list, columns=quants)
    stats = {}
    for model_id in model_list:
        stats[model_id] = {
            "latency": [],
            "duration": [],
            "memory": [],
        }

        for quant in quants:
            model_dir = f"/data/{model_id}-{quant}"
            print(f"Loading model: {model_id} with {quant} quantization")
            if quant == "BF16":
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                memory = model_memory_footprint(model)
            elif quant == "AWQ":
                model = AutoAWQForCausalLM.from_pretrained(
                    model_dir,
                    safetensors=True,
                    trust_remote_code=True,
                    device_map="auto",
                )
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                memory = model_memory_footprint_awq(model)
            elif quant == "GPTQ":
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    device_map="auto",
                )
                tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
                memory = model_memory_footprint(model)
            else:
                break

            result = generate_helper(model, tokenizer, prompt)
            release_memory(model)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            response_df.loc[model_id, quant] = [
                value for key, value in result.items() if key == "text"
            ]
            stats[model_id]["latency"].append(result["latency"])
            stats[model_id]["duration"].append(result["duration"])
            stats[model_id]["memory"].append(memory)

    save_plot(stats)
