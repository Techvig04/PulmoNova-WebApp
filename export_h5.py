# export_h5.py
import os
from tensorflow.keras.models import load_model

# Path to your current model artefact
# If you have a folder `models/my_model.keras`, point there.
orig_path = os.path.join('models', 'my_model.keras')

# Load it just to re-serialize
model = load_model(orig_path)

# Now save as HDF5
h5_path = os.path.join('models', 'my_model.h5')
model.save(h5_path, save_format='h5')

print("Saved HDF5 model to", h5_path)
