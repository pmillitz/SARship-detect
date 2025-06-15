import os
from vispy import app
from visualise_labels import SARFish_Plot
import numpy as np

# Force software rendering (optional)
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

# Downsampled image
data = np.random.rand(555, 844).astype(np.float32)
nodata_mask = np.random.randint(0, 2, (555, 844), dtype=np.uint8)

# Use EGL backend (if not set already)
os.environ["VISPY_APP_BACKEND"] = "egl"
os.environ["VISPY_GL_BACKEND"] = "pyopengl"

# Initialize the plot
plot = SARFish_Plot(data, nodata_mask, title="Standalone Test", show=True)

# Run event loop
app.run()
