from .Chat import Chat
from .Logger import Logger
from .prompt import PROMPT
import time
import torch
from typing import Literal, Tuple
import gc

IMPLEMENTED_METHODS = Literal['genetic', 'rs']

class AdversarialAttack():
  def __init__(
    self,
    chat:Chat=None, 
    logger:Logger=None,
    population_size:int=10,
    mutation_probability:int=0.01,
    stop_criterion:int=1000,
    adv_suffix_length:int=100,
    batch_size:int=10,
    elitism_percentage:int=0.07,
    rs_token_to_change:int=4,
    target_token:str="Sure"
  ) -> None:
    if logger: assert logger.is_initialized(), "Logger not initialized."
    
    self.chat = chat
    self.logger = logger
    self.population_size = population_size
    self.mutation_probability = mutation_probability
    self.stop_criterion = stop_criterion
    self.target_token = target_token
    self.target_id = chat.tokenize(self.target_token).item()
    self.adv_suffix_length = adv_suffix_length
    self.elitist_individuals = int(self.population_size * elitism_percentage)
    self.batch_size = batch_size
    self.rs_token_to_change = rs_token_to_change
    self.best_individuals = None
    self._population = None
    self._GENETIC_METHOD = 'genetic'
    self._RANDOM_SEARCH_METHOD = 'rs'
    self.METHODS = [self._GENETIC_METHOD, self._RANDOM_SEARCH_METHOD]
  
  def create_population(self) -> torch.Tensor:
    vocab_size = self.chat.tokenizer.vocab_size
    population = torch.randint(0, vocab_size, (self.population_size,self.adv_suffix_length))
    return population
  
  def evaluate_fitness(self, population:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
    max_new_tokens = 1
    population_str = self.chat.detokenize(population)
    prompts = [PROMPT + adv_tokens for adv_tokens in population_str]
    
    responses = []
    fitness = torch.Tensor()
    iteration = 0
    print(f"> Batch generation[{len(prompts) // self.batch_size}]:", end=" ")
    
    # batch evaluation
    for i in range(0, len(prompts), self.batch_size):    # makes a larger population possible
      iteration += 1
      print(iteration, end=" ")
      prompts_batch = prompts[i:i+self.batch_size]
      _responses, _probs = self.chat.ask(prompts_batch, max_new_tokens = max_new_tokens)
      _probs = _probs.to('cpu')
      responses = responses + [response[len(prompts_batch[j]):] for j, response in enumerate(_responses)]
      fitness = torch.concat((fitness, _probs[0, :, self.target_id]), dim=0)
      gc.collect()
      torch.cuda.empty_cache()
    
    print(".")

    fitness = fitness + 1e-16
    probs = fitness / fitness.sum()
    prompts = None
    fitness = fitness.to('cpu')
    probs = probs.to('cpu')
    return fitness, probs, responses
  
  def selection_function(self, probs: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
    individuals_size = len(probs) + len(probs)%2
    individuals = torch.multinomial(probs, num_samples=individuals_size, replacement=True)
    pairs = torch.reshape(individuals, (-1, 2)).to('cpu')
    pairs = population[pairs]
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
    
    for i in range(0, self.stop_criterion):
      start = time.time()
      print(f"> Step {i + 1}")
      fitness, probs, responses = self.evaluate_fitness(population)
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
          "fitness_std": fitness.std().item(),
          "sure_count": responses.count(self.target_token)
        }
        self.logger.log(log_values)
      print(f"> Responses: {responses}")
      print(f"> Best fitness: {torch.max(fitness)}")
      print(f"> Best probs: {torch.max(probs)}")
      individuals_pairs = self.selection_function(probs, population)
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
    
    for i in range(0, self.stop_criterion):
      start = time.time()
      print(f"> ITERATION {i + 1}")  
      offspring = self.random_search_step(population)
      fitness_2, _, responses_2 = self.evaluate_fitness(offspring)
      if self.logger != None:
        for index, fitness in enumerate(fitness_2):
          log_values = {
            "run": index + 1,
            "iteration": i,
            "fitness": fitness.item(),
            "sure_generated": int(responses_2[index] == self.target_token)
          }
          self.logger.log(log_values)
      end = time.time()
      print(f"> New individuals: {(fitness_2 > fitness_1).sum()} / {self.population_size}")
      print(f"> Best fitness: {torch.max(fitness_1)} | {torch.max(fitness_2)}")
      print(f"> Responses: {responses_2}")
      print(f"> Cycle took {end - start}s...")
      population[fitness_2 >= fitness_1] = offspring[fitness_2 >= fitness_1]
      fitness_1[fitness_2 >= fitness_1] = fitness_2[fitness_2 >= fitness_1]
      print("\n ---------- \n")
      
    self.best_individuals = population
      
  def run(self, method:IMPLEMENTED_METHODS, run:int=0):
    assert method in self.METHODS, f"Method not valid, use one of {self.METHODS}"
    
    if method == self._GENETIC_METHOD: self.genetic_algorithm(run=run)
    if method == self._RANDOM_SEARCH_METHOD: self.random_search()
