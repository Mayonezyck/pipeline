import yaml
import numpy as np

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        print(f'The file {file_path} is not found.')
    except yaml.YAMLError as error:
        print(f'Having problem loading the YAML file: {error}')

def save_yaml(file_path, data):
    try:
        print(data)
        with open(file_path, 'w') as file:
            for item in data:
                file.write(f'{item}\n')
    except yaml.YAMLError as error:
        print(f'Having problem saving the YAML file: {error}')