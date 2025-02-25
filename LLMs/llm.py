from openai import OpenAI
from typing import Union, Dict, List, Optional

class LLM:
    def __init__(self, api_key, api_base_url, model_name):
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        pass
    
    def run(self, messages, max_new_tokens=4096, temperature=0.2, return_num_responses=1):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=return_num_responses,  # Generate 3 responses
            messages=messages,
        )
        if return_num_responses==1:
            return completion.choices[0].message.content
        else:
            return [choice.message.content for choice in completion.choices]
        
        