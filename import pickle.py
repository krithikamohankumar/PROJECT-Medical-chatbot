import pickle
with open('knn(2).pkl', 'rb') as f:
    # Load the pickled object
    loaded_object = pickle.load(f)

# Now you can work with the loaded object
print(loaded_object)