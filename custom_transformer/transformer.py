import argparse
import base64
import io
from io import BytesIO
from typing import Dict, Union

import numpy as np
from kserve import (
    InferInput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
    model_server,
)
from kserve.model import PredictorProtocol
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse
from PIL import Image
from torchvision import transforms


def image_transform(data):
    preprocess = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    byte_array = base64.b64decode(data)
    image = Image.open(io.BytesIO(byte_array))
    tensor = preprocess(image)
    return tensor


# def image_transform(model_name, data):
#     """converts the input image of Bytes Array into Tensor
#     Args:
#         model_name: The model name
#         data: The input image bytes.
#     Returns:
#         numpy.array: Returns the numpy array after the image preprocessing.
#     """
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     if model_name == "mnist" or model_name == "cifar10":
#         preprocess = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#         )
#     byte_array = base64.b64decode(data)
#     image = Image.open(io.BytesIO(byte_array))
#     tensor = preprocess(image).numpy()
#     return tensor


class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.ready = True

    def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:
        instances = payload["instances"]
        processed_instances = list()
        for ins in instances:
            base64_string = ins.split(",")[1].strip()
            image_bytes = base64.b64decode(base64_string)
            image_file = BytesIO(image_bytes)

            img = Image.open(image_file)
            img = img.convert("L").resize((28, 28))
            nparr = np.asarray(img)
            nparr = nparr.reshape((28, 28, 1))
            nparr = nparr.reshape(1, 28, 28, 1)

            processed_instances.append(nparr.tolist())

        return {"instances": processed_instances}

    # def postprocess(
    #     self,
    #     infer_response: Union[Dict, ModelInferResponse],
    #     headers: Dict[str, str] = None,
    # ) -> Union[Dict, InferResponse]:
    #     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function", required=True
    )
    # parser.add_argument("--model_name", help="The name that the model is served under.")
    args, _ = parser.parse_known_args()

    model = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    ModelServer().start([model])
