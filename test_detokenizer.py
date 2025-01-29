import pandas as pd
import csv
from tqdm import tqdm
from utils.Chat import Chat
from utils.prompt import PROMPT

# loading the model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"

device = 'cuda'
quantized = False
chat = Chat(model_name, device=device, quantized=quantized)


# loading the dataset
suffixes = pd.read_csv("./data/MERGED-llama3.0-8B-suffix.csv")

file_name = "./log/detokenized_8B.csv"
records = []

with open(file_name, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["type", "N", "run", "suffix", "string"])
    writer.writeheader()

    for i, suffix in enumerate(suffixes.values):
        type, N, run, suffix = suffix
        suffix_eval = eval(suffix.replace("|", ","))
        adv_suffix = chat.detokenize([suffix_eval])[0]
        record = [{
            "type": type,
            "N": N,
            "run": run,
            "suffix": suffix,
            "string": adv_suffix
        }]
        writer.writerows(record)
        file.flush()
