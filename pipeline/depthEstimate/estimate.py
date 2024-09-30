import os
def estimate(method, path):
    try:
        if method == 'depth-anything':
            pass
        return 
    except FileNotFoundError:
        print(f"The directory {path} does not exist.")
        return 
    except Exception as e:
        print(f"An error occurred: {e}")
        return 