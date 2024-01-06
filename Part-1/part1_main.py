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

from torch.profiler import profile, ProfilerActivity
from utils import save_plot

MODEL_LIST= [
"Salesforce/codegen2-1B",
"Salesforce/codegen2-3_7B",
"Salesforce/codegen2-7B",
]

QUANT_TYPE = [ "FP32", "BF16", "INT8", "INT4", "NF4"]


def model_memory_footprint(model):
    return float(f"{model.get_memory_footprint() / 1024 / 1024 / 1024:.2f}")


def generate_helper(model, tokenizer, prompt,trace_filename):
    
    inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to("cuda")
    
    # warm up
    for i in range(3):
        _ = model.generate(inputs, max_length=15, pad_token_id=tokenizer.eos_token_id)

    start = time.perf_counter()
    generated_tokens = model.generate(
        inputs,
        max_length=48,
        pad_token_id=tokenizer.eos_token_id
    )
    duration = time.perf_counter() - start
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    generated_text = generated_text[len(prompt) :]

    latency_per_token_in_ms = (
        duration / len(generated_tokens[0])
    ) * 1000
    
    return {
        "text": generated_text,
        "latency": float(f"{round(latency_per_token_in_ms,2)}"),
        "duration": duration,
    }

if __name__ == "__main__":
    
    prompt = "def heap_sort(x):"

    response_df = pd.DataFrame(index=MODEL_LIST, columns=QUANT_TYPE)
    stats = {}
    for model_id in MODEL_LIST:
        stats[model_id] = {
            "latency": [],
            "duration": [],
            "memory": [],
        }
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        for quant in QUANT_TYPE:
            if quant == "FP32":
                kwargs = {"torch_dtype": torch.float32}
            elif quant == "BF16":
                kwargs = {"torch_dtype": torch.bfloat16}
            elif quant == "INT8":
                eight_bit_config = BitsAndBytesConfig(load_in_8bit=True, load_in_8bit_fp32_cpu_offload=True)
                kwargs = {"quantization_config": eight_bit_config}
            elif quant == "INT4":
                four_bit_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                )
                kwargs = {"quantization_config": four_bit_config}
            elif quant == "NF4":
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                kwargs = {"quantization_config": nf4_config}

            print(f"Loading model: {model_id} with {quant} quantization")

            if model_id == "Salesforce/codegen2-7B":
                kwargs['offload_folder'] = "/data/offload"
                kwargs['low_cpu_mem_usage'] = True
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                revision="main",
                device_map="auto",
                **kwargs,
            )
            memory = model_memory_footprint(model)


            trace_filename = f'trace-{model_id.replace("/", "-")}-{quant}.json'
            result = generate_helper(model, tokenizer, prompt, trace_filename)
            # clear_memory(model, tokenizer)
            release_memory(model)
            del model
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