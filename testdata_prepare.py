from io import BytesIO
from random import randint

import torchvision
from PIL import Image


def image_to_bytes(img: Image):
    """Download an test image from MNIST dataset to use.

    No image tranform is applied at this stage.
    """
    testset = torchvision.datasets.MNIST(
        "dataset/", train=False, download=True, transform=None
    )
    image_to_use, tag = testset[randint(0, 9999)]
    print(f"Use image of digit {tag}")
    buffered = BytesIO()
    image_to_use.save(
        buffered, format="PNG"
    )  # Specify the desired format (e.g., "PNG", "JPEG")
    return buffered.getvalue()
