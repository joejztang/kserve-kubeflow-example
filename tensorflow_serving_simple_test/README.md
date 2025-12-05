# tensorflow simple test

1. Prepare data. We are testing against a customized trained digit classifier, so naturally need to massage the data from image to numpy array first. Use pillow and numpy will do the work.

2. Setup tensorflow server. We can follow [Serve a tensorflow model in 60 seconds](https://github.com/tensorflow/serving?tab=readme-ov-file#serve-a-tensorflow-model-in-60-seconds) to test, specifically in local for me was `docker run -t --rm -p 8501:8501 -v "./models:/models" -e MODEL_NAME=mnist_tf tensorflow/serving`. Make sure tho you have models folder in the same folder/workspace you are working in.

3. Simple curl call. For me I prepared my data in `1.json` using the `data_prepare.py` script. Then I do things similar to `curl -d @tensorflow_simple_test/1.json -X POST http://localhost:8501/v1/models/mnist_tf:predict`. You can expect similiar response with `{"predictions": [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}`