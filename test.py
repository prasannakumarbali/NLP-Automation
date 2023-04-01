# import pickle

# # Open the pkl file in read binary mode
# with open('model1.pkl', 'rb') as f:
#     # Load the data from the pkl file
#     data = pickle.load(f)

# # Convert the loaded data to a string
# data_str = str(data)

# print(data_str)
import pickle

# open the PKL file and load the Python object
with open('model1.pkl', 'rb') as f:
    data = pickle.load(f)

# print the Python object
print(data)