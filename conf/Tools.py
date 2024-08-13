import json

class Tools:
    
    def read_json(file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            return json.load(f)
    
    def write_json(data, save_path):
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(data, file_path, ensure_ascii=False)