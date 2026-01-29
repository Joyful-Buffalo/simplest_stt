import yaml
import os

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"YAML file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def load_proj_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))