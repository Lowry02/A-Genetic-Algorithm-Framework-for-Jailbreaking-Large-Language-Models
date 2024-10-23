from utils.AdversarialAttack import AdversarialAttack
from utils.Logger import Logger
from utils.Chat import Chat
from argparse import ArgumentParser
import torch
import gc
import os

# genetic algorithm
parser = ArgumentParser()
parser.add_argument("-m", "--model_name", dest="model_name", help="Model name to use")
parser.add_argument("-l", "--log_name", dest="log_name", help="Log file name")
parser.add_argument("-b", "--batch_size", dest="batch_size", help="Generator batch size", type=int, default=10)
args = parser.parse_args()
model_name = args.model_name
file_name = args.log_name
batch_size = args.batch_size

print(f"> GENETIC ALGORITHM[{model_name}]\n")
print("> Loading the model...")

log_folder = "log"
if log_folder not in os.listdir():
  os.mkdir(log_folder)

torch.cuda.empty_cache()

header = [
  "run",
  "iteration",
  "fitness_max",
  "fitness_min",
  "fitness_median",
  "fitness_mean",
  "fitness_std",
  "sure_count"
]
logger = Logger(header=header)
logger.create_file(f"./log/{file_name}.csv")

suffix_header = ["run", "suffix"]
suffix_logger = Logger(header=suffix_header)
suffix_logger.create_file(f"./log/{file_name}-suffix.csv")
device = 'cuda'
quantized = False
chat = Chat(model_name, device=device, quantized=quantized)

population_size = 50
mutation_probability = 0.08
elitism_percentage = 0.07
stop_criterion = 1000
adv_suffix_length = 25

print("> Attack starting...")

attacker = AdversarialAttack(
  chat=chat,
  logger=logger,
  population_size=population_size,
  mutation_probability=mutation_probability,
  stop_criterion=stop_criterion,
  adv_suffix_length=adv_suffix_length,
  elitism_percentage=elitism_percentage,
  batch_size=batch_size
)

best_individuals = []

for run in range(1,31):
  print(f"> Run {run}")
  attacker.run(method='genetic', run=run)
  gc.collect()
  torch.cuda.empty_cache()
  suffix_logger.log({"run": run, "suffix": str(attacker.best_individuals.tolist()).replace(",", "|")})

logger.close_file()
suffix_logger.close_file()

print("> END")

