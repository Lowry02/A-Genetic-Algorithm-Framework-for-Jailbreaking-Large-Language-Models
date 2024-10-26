"""_summary_
#### Description
Class used to connect with an huggingface model.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding, BitsAndBytesConfig, GenerationConfig
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
from dotenv import load_dotenv
import os

class Chat:
  def __init__(self, model_name: str, device:str='cpu', quantized:bool=True) -> None:
    assert isinstance(model_name, str), "model_name must be a string"
    assert isinstance(device, str), "device must be a string"
    
    load_dotenv()
    self.model_name = model_name
    self.device = device
    token = os.getenv("HF_TOKEN")
    cache_dir = os.getenv("CACHE_DIR")
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token, cache_dir=cache_dir, padding_side="left")
    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # to avoid an error
    # self.generation_config = GenerationConfig.from_pretrained(model_name, token=token, cache_dir=cache_dir)
    
    if quantized:
      quantization_config = BitsAndBytesConfig(load_in_8bit=quantized)
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=quantization_config, device_map='auto', token=token, cache_dir=cache_dir)
    else:
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto', token=token, cache_dir=cache_dir)
    
  def ask(self, prompt:str | list, max_new_tokens:int=10) -> Union[str|list, torch.Tensor]:
    """_summary_
    #### Description
    Used to chat with the bot.
    
    Args:
        prompt (str | list): list or string of prompts
        max_new_tokens (int, optional): Max tokens that the model can generate. Defaults to 10.

    Returns:
        Union[str|list, torch.Tensor]: returns the response for each prompt and the generation probabilitiy of every token at each generation.
    """
    ## Return
    #  - response: string that contains the generated text
    #  - output.scores: list containing the scores of every token in the vucabulary for each model generation
    assert isinstance(prompt, str) or isinstance(prompt, list), "prompt must be a string or a list"
    assert isinstance(max_new_tokens, int), "max_new_tokens must be an int"
    
    with torch.no_grad():
      if isinstance(prompt, str):
        messages = [{"role": "user", "content": f"{prompt}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        messages = [
          [{"role": "user", "content": f"{string}"}]
          for string in prompt
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

      encoded_input = self.tokenizer(prompt, padding=True, return_tensors='pt', add_special_tokens=False).to(self.device)
  
      output = self.model.generate(
        **encoded_input,
        # generation_config=self.generation_config,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=self.tokenizer.eos_token_id,
      )
      
      prompt = self.tokenizer.batch_decode(encoded_input['input_ids'], skip_special_tokens=True)
      responses = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)

      responses = [response[len(prompt[j]):] for j, response in enumerate(responses)]
      output.scores = torch.stack(output.scores)    # generated_tokens x batch_size x scores
      probs = nn.Softmax(dim=2)(output.scores)
        
    return responses, probs
  
  def tokenize(self, string:str) -> BatchEncoding:
    """_summary_
    #### Description
    Used to convert a string to tokens
    
    Args:
        string(str)
    Returns:
        _type_: returns the tokenization of the string
    """
    assert isinstance(string, str), "string must be a string"
  
    return self.tokenizer.encode(string, return_tensors='pt', add_special_tokens=False)
  
  def detokenize(self, ids: Tensor | list) -> str:
    """_summary_

    Args:
        ids (Tensor | list): list of token ids

    Returns:
        str: the string obtained from the tokenization of the ids
    """
    assert isinstance(ids, Tensor) or isinstance(ids, list), "ids must be a Tensor or a list"

    return self.tokenizer.batch_decode(ids, skip_special_tokens=False)
