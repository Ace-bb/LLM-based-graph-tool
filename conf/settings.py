from dotenv import load_dotenv
load_dotenv()
import json, os
from rich import console
from datetime import datetime
from .Tools import Tools

tools = Tools()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = console.Console()
print = logger.log

def load_config(model_name):
    file_path = "./conf/config.json"
    with open(file_path, "r") as file:
        config = json.load(file)
    if model_name in ['gpt-4o', 'gpt-4o-mini']:
        config['base_url'] = os.environ['OPENAI_BASE_URL']
        config['api_key'] = "sk-pYG3maJnnIkvB8ct74303dEdBb6c446bB96d7c9f2bD1C3Ff" # os.environ['OPENAI_BASE_KEY']
    elif model_name in ['qwen-plus', 'qwen-vl-plus-latest', 'qwen2.5-vl-72b-instruct', 'qwen2.5-vl-7b-instruct', 'qwen2.5-vl-3b-instruct', 'llama3.2-90b-vision-instruct', 'llama3.2-11b-vision']:
        config['base_url'] = os.environ['DASHSCOPE_BASE_URL']
        config['api_key'] = os.environ['DASHSCOPE_API_KEY']
    elif model_name in ['internvl2.5-latest']:
        config['base_url'] = "https://chat.intern-ai.org.cn/api/v1"
        config['api_key'] = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI0MDA1Nzg5Iiwicm9sIjoiUk9MRV9SRUdJU1RFUiIsImlzcyI6Ik9wZW5YTGFiIiwiaWF0IjoxNzQwODUwMzM4LCJjbGllbnRJZCI6ImVibXJ2b2Q2eW8wbmx6YWVrMXlwIiwicGhvbmUiOiIxMzQ4OTMyMzI4NSIsInV1aWQiOiIzNTYwYjNlNy1jZDQ0LTRiOGItYTg5Mi0yMDFiYWRkNTBhMzIiLCJlbWFpbCI6IjEzNDg5MzIzMjg1QDE2My5jb20iLCJleHAiOjE3NTY0MDIzMzh9.PLx_fdc-3rnM5ZTs3f_oJnsTF-bV3-1Aj4QUo6rEYfy7tPt0CRIFXSDSejdTVqy3um24IHOF7mNqT_7iJ05lPA"
    elif model_name in ['internvl2.5-local', 'internvl2.5-8B']:
        config['base_url'] = "http://0.0.0.0:23333/v1"
        config['api_key'] = "1234567890"
    return config
