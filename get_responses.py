from utils.Chat import Chat
from utils.Logger import Logger
from utils.prompt import PROMPT
import pandas as pd
import ast

def get_adv_questions(file_name:str, chat:Chat) -> list[str]:
  data = pd.read_csv(file_name)
  data['run'] = data['run'].astype(int)
  data['suffix'] = data['suffix'].astype(str)

  adv_suffixes = []
  for suffix in data['suffix']:
    suffix = suffix.replace("|", ',')
    suffix = ast.literal_eval(suffix)
    adv_suffix = chat.detokenize([suffix])
    adv_suffixes.append(PROMPT + adv_suffix[0])

  return adv_suffixes

def log_responses(input_name:str, output_name:str, chat:Chat) -> None:
  adv_questions = get_adv_questions(input_name, chat)
  max_new_tokens = 500

  responses = []

  i = 1
  for prompt in adv_questions:
    print(f"Generating {i}")
    i += 1
    response, _ = chat.ask(prompt, max_new_tokens=max_new_tokens)
    responses.append(response[0])

  logger = Logger(header=['number', 'response'])
  logger.create_file(output_name)

  for i, response in enumerate(responses):
    response = response.replace('\n', '\\n')
    logger.log({"number": i, "response": f"\"{response}\""})

  logger.close_file()

# main
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# model_name = "meta-llama/Llama-3.2-3B-Instruct"
device = 'cuda'
quantized = False
chat = Chat(model_name, device=device, quantized=quantized)

#log_responses(input_name="./rs-llama3.0-8B-suffix.csv", output_name="./rs-responses.csv", chat=chat)
log_responses(input_name="./log/llama3.2-3B-suffix.csv", output_name="./log/ga-responses-8B.csv", chat=chat)
