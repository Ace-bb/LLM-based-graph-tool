from dotenv import load_dotenv
load_dotenv()
import json, os
from rich import console
from datetime import datetime
from .Tools import Tools

tools = Tools()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = console.Console()

def load_config(model_name):
    file_path = "./conf/config.json"
    with open(file_path, "r") as file:
        config = json.load(file)
    if model_name in ['gpt-4o']:
        config['base_url'] = os.environ['OPENAI_BASE_URL']
        config['api_key'] = os.environ['OPENAI_BASE_KEY']
    elif model_name in ['qwen2']:
        config['base_url'] = os.environ['DASHSCOPE_BASE_URL']
        config['api_key'] = os.environ['DASHSCOPE_API_KEY']
    return config
