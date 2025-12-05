import json

import numpy as np
from PIL import Image

img = Image.open("<your_mnist_png_path>")
img = img.convert("L").resize((28, 28))
nparr = np.asarray(img)
nparr = nparr.reshape((28, 28, 1))
nparr = nparr.reshape(1, 28, 28, 1)
json.dumps(dict(instances=nparr.tolist()))
