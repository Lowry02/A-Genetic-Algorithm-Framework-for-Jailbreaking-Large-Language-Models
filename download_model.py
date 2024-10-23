from argparse import ArgumentParser
from utils.Chat import Chat

parser = ArgumentParser()
parser.add_argument("-m", "--model_name", dest="model_name", help="Model name to use")
args = parser.parse_args()
model_name = args.model_name

print(f"> Downloading {model_name}")
quantized = False
chat = Chat(model_name, quantized=quantized)
print("> End")