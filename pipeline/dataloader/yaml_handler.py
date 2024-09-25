import yaml

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        print(f'The file {file_path} is not found.')
    except yaml.YAMLError as error:
        print(f'Having problem loading the YAML file: {error}')

