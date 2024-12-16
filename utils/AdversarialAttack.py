from .Chat import Chat
from .Logger import Logger
from .prompt import PROMPT
import time
import torch
from typing import Literal, Tuple
import gc
import json
import pandas as pd

IMPLEMENTED_METHODS = Literal['genetic', 'rs']

class AdversarialAttack():
  def __init__(
    self,
    chat:Chat=None, 
    logger:Logger=None,
    population_size:int=10,
    mutation_probability:int=0.01,
    tournament_size:int=2,
    stop_criterion:int=1000,
    adv_suffix_length:int=100,
    batch_size:int=10,
    elitism_percentage:int=0.07,
    rs_token_to_change:int=4,
    target_token:str="Sure",
    save_steps:int=10,
    checkpoint_path:str=None,
    resume:bool=False,
    n_prompt:int=1
  ) -> None:
    if logger: assert logger.is_initialized(), "Logger not initialized."
    
    self.chat = chat
    self.logger = logger
    self.population_size = population_size
    self.mutation_probability = mutation_probability
    self.tournament_size = tournament_size 
    self.stop_criterion = stop_criterion
    self.target_token = target_token
    self.target_id = None if chat == None else chat.tokenize(self.target_token).item()
    self.adv_suffix_length = adv_suffix_length
    self.elitist_individuals = int(self.population_size * elitism_percentage)
    self.batch_size = batch_size
    self.rs_token_to_change = rs_token_to_change
    self.best_individuals = None
    self._population = None
    self._GENETIC_METHOD = 'genetic'
    self._RANDOM_SEARCH_METHOD = 'rs'
    self.METHODS = [self._GENETIC_METHOD, self._RANDOM_SEARCH_METHOD]
    self.checkpoint_path = checkpoint_path
    self.save_steps = save_steps

    if resume:
      with open(checkpoint_path, 'r') as f:
        self.checkpoint_data = json.load(f)
    else:
      self.checkpoint_data = None

    ds_path = "../data/harmful_behavior.csv"
    df = pd.read_csv(ds_path)
    self.base_prompts = []
    for _, question in df.head(n_prompt).iterrows():
      question_id, goal, target_answer = question
      self.base_prompts.append(PROMPT.format(goal=goal, target_answer=target_answer))
  
  def create_population(self) -> torch.Tensor:
    if self.checkpoint_data:
      population = torch.tensor(self.checkpoint_data.population)
    else:
      vocab_size = self.chat.tokenizer.vocab_size
      population = torch.randint(0, vocab_size, (self.population_size,self.adv_suffix_length))
    return population
  
  def save_checkpoint(self, iteration, population):
    checkpoint = {
      "iteration" : iteration,
      "population" : population.tolist()
    }
    with open(self.checkpoint_path, "w") as f:
      json.dump(checkpoint, f)
  
  def evaluate_fitness(self, population:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
    max_new_tokens = 1
    population_str = self.chat.detokenize(population)
    all_prompts = []
    for base_prompt in self.base_prompts:
      all_prompts.append([base_prompt + adv_tokens for adv_tokens in population_str])

    pop_size = len(all_prompts[0])
    fitnesses = torch.zeros((len(all_prompts), pop_size))
    n_responses = []

    for idx, prompts in enumerate(all_prompts):
      responses = []
      prompt_fitness = torch.Tensor()
      iteration = 0
      print(f"> Prompt {idx+1}: batch generation[{len(prompts) // self.batch_size}]:", end=" ")
      
      # batch evaluation
      for i in range(0, pop_size, self.batch_size):    # makes a larger population possible
        iteration += 1
        print(iteration, end=" ")
        prompts_batch = prompts[i:i+self.batch_size]
        _responses, _probs = self.chat.ask(prompts_batch, max_new_tokens = max_new_tokens)
        _probs = _probs.to('cpu')
        responses = responses + [response for response in _responses]
        prompt_fitness = torch.concat((prompt_fitness, _probs[0, :, self.target_id]), dim=0)
        gc.collect()
        torch.cuda.empty_cache()
      fitnesses[idx] = prompt_fitness
      n_responses.append(responses)
    
    print(".")

    fitness = fitnesses.sum(dim=0)
    fitness = fitness + 1e-16
    probs = fitness / fitness.sum() #probs for tournament selection
    prompts = None
    fitness = fitness.to('cpu')
    probs = probs.to('cpu')
    return fitness, probs, fitnesses, n_responses
  
  def roulette_wheel_selection(self, probs: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
    individuals_number = len(probs) + len(probs)%2
    individuals = torch.multinomial(probs, num_samples=individuals_number, replacement=True)
    pairs = torch.reshape(individuals, (-1, 2)).to('cpu')
    pairs = population[pairs]
    return pairs
  
  def tournament_selection(self, probs: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
    individuals_number = len(probs) + len(probs)%2
    individuals_idxs = torch.randint(0, len(probs), size=(individuals_number, self.tournament_size))
    tournament_individuals = population[individuals_idxs]
    tournament_probs = probs[individuals_idxs]
    tournament_best = torch.argmax(tournament_probs, dim=1)
    tournament_individuals = tournament_individuals[range(individuals_number), tournament_best]
    pairs = torch.reshape(tournament_individuals, (-1, 2, self.adv_suffix_length)).to('cpu')
    return pairs
    
  def crossover_function(self, individuals_pairs: torch.Tensor) -> torch.Tensor:
    split_point = torch.randint(1, self.adv_suffix_length - 1, (1,)).item()
    children_a = torch.concat((individuals_pairs[:, 0, :split_point], individuals_pairs[:, 1, split_point:]), dim=1)
    children_b = torch.concat((individuals_pairs[:, 1, :split_point], individuals_pairs[:, 0, split_point:]), dim=1)
    offspring = torch.cat((children_a, children_b), dim=0)
    return offspring
  
  def mutation_function(self, offspring: torch.Tensor):
    vocab_size = self.chat.tokenizer.vocab_size
    mutation_mask = torch.full(offspring.shape, self.mutation_probability)
    mutation_mask = torch.bernoulli(mutation_mask).bool()
    mutation_values = torch.randint(0, vocab_size, (mutation_mask.sum(), ))
    offspring[mutation_mask] = mutation_values
    return offspring
  
  def update_population(self, population:torch.Tensor, offspring:torch.Tensor, probs:torch.Tensor):
    deleted_children = torch.randint(0, len(offspring), (self.elitist_individuals,))
    _, best_individuals = torch.topk(probs, k=self.elitist_individuals)
    offspring[deleted_children] = population[best_individuals]
    return offspring
  
  def random_search_step(self, population: torch.Tensor) -> torch.Tensor:
    # used to generate the new items
    indices = torch.rand(population.shape).argsort(dim=1)
    mask = torch.zeros(population.shape, dtype=torch.bool)
    mask.scatter_(1, indices[:, :self.rs_token_to_change], True)
    offspring = population.detach().clone()
    offspring[mask] = torch.randint(0, self.chat.tokenizer.vocab_size, (self.rs_token_to_change * population.size(0), ))
    return offspring

  def genetic_algorithm(self, run:int=1):
    self.best_individuals = []
    population = self.create_population()
    starting_generation = self.checkpoint_data.iteration if self.checkpoint_data else 0

    for i in range(starting_generation, self.stop_criterion):
      if i % self.save_steps == 0:
        self.save_checkpoint(i, population)
      start = time.time()
      print(f"> Step {i + 1}")
      fitness, probs, fitnesses, responses = self.evaluate_fitness(population)
      best_index = fitness.argmax()
      self.best_individuals = population[best_index]
      if self.logger != None:
        log_values = {
          "run": run,
          "iteration": i,
          "fitness_max": fitness.max().item(),
          "fitness_min": fitness.min().item(),
          "fitness_median": fitness.median().item(),
          "fitness_mean": fitness.mean().item(),
          "fitness_std": fitness.std().item()
        }
        for prompt_id in range(len(self.base_prompts)):
          log_values.update(
            {
              f"fitness_max_{prompt_id}": fitnesses[prompt_id].max().item(),
              f"fitness_min_{prompt_id}": fitnesses[prompt_id].min().item(),
              f"fitness_median_{prompt_id}": fitnesses[prompt_id].median().item(),
              f"fitness_mean_{prompt_id}": fitnesses[prompt_id].mean().item(),
              f"fitness_std_{prompt_id}": fitnesses[prompt_id].std().item(),
              f"sure_count_{prompt_id}": responses[prompt_id].count(self.target_token)
            }
          )
        self.logger.log(log_values)
      print(f"> Responses: {responses[0]}")
      print(f"> Best fitness: {torch.max(fitness)}")
      print(f"> Best probs: {torch.max(probs)}")
      individuals_pairs = self.tournament_selection(probs, population)
      offspring = self.crossover_function(individuals_pairs)
      offspring = self.mutation_function(offspring)
      population = self.update_population(population, offspring.detach().clone(), probs)
      end = time.time()
      print(f"> Cycle took {end - start}s...")
      gc.collect()
      torch.cuda.empty_cache()
      
  def random_search(self):
    self.best_individuals = []
    population = self.create_population()
    fitness_1 = torch.zeros((population.size(0), ))
    starting_iteration = self.checkpoint_data.iteration if self.checkpoint_data else 0
    
    for i in range(starting_iteration, self.stop_criterion):
      if i % self.save_steps == 0:
        self.save_checkpoint(i, population)
      start = time.time()
      print(f"> ITERATION {i + 1}")  
      offspring = self.random_search_step(population)
      fitness_2, _, fitnesses_2, responses_2 = self.evaluate_fitness(offspring)
      if self.logger != None:
        for index, fitness in enumerate(fitness_2):
          log_values = {
            "run": index + 1,
            "iteration": i,
            "fitness": fitness.item()
          }
          for prompt_id in range(len(self.base_prompts)):
            log_values.update(
              {
                f"fitness_{prompt_id}": fitnesses_2[prompt_id, index].item(),
                f"sure_generated_{prompt_id}": int(responses_2[prompt_id, index] == self.target_token)
              }
            )
          self.logger.log(log_values)
      end = time.time()
      print(f"> New individuals: {(fitness_2 > fitness_1).sum()} / {self.population_size}")
      print(f"> Best fitness: {torch.max(fitness_1)} | {torch.max(fitness_2)}")
      print(f"> Responses: {responses_2[0]}")
      print(f"> Cycle took {end - start}s...")
      population[fitness_2 >= fitness_1] = offspring[fitness_2 >= fitness_1]
      fitness_1[fitness_2 >= fitness_1] = fitness_2[fitness_2 >= fitness_1]
      print("\n ---------- \n")
      
    self.best_individuals = population
      
  def run(self, method:IMPLEMENTED_METHODS, run:int=0):
    assert method in self.METHODS, f"Method not valid, use one of {self.METHODS}"
    
    if method == self._GENETIC_METHOD: self.genetic_algorithm(run=run)
    if method == self._RANDOM_SEARCH_METHOD: self.random_search()
