import pickle


# Function to load a pickle object from a file
def load_pickle(filename):
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"Error loading pickle object from {filename}: {str(e)}")
        return None


# Function to save a pickle object to a file
def save_pickle(data, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Pickle object saved to {filename}")
    except Exception as e:
        print(f"Error saving pickle object to {filename}: {str(e)}")
