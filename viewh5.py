import h5py
from tensorflow.keras.models import load_model

# Load the .h5 model
model = load_model('model/trained_model.h5')

# Print the model architecture
model.summary()
# Open the .h5 file
h5_file = h5py.File('model/trained_model.h5', 'r')

# List all groups in the HDF5 file (like weights and layers)
print("Keys in the HDF5 file:", list(h5_file.keys()))

# Explore details (for example, view the structure of 'model_weights')
weights_group = h5_file['model_weights']
print("Layers in model_weights:", list(weights_group.keys()))

# Close the file after inspection
h5_file.close()