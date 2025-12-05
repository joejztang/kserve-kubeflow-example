# kserve simple test

In order to correctly setup kserve locally, you will need to do some prerequisites:

1. Having `secret.yaml`. Replace your gcp sa generated either from console or systematically, with correct role bound. For simplicity, I just give `Storage Admin` for my little test.

2. Having `serviceaccount.yaml`. Apply both secret and serviceaccount using `kubectl apply -f ...`

3. Copy what's in `endpoint.yaml`, and go to kubeflow portal, then go to `Kserve Enpoints`. `New Endpoint` and paste what's in `endpoint.yaml`. Finally Create.

# Test

1. port forward kserve pod, container port 8080 to your favorite local port. (I chose 8081)

2. Do similar things with `curl -d @tensorflow_simple_test/1.json -X POST -H "Content-Type: application/json" http://localhost:8081/v1/models/mnist-tf:predict`.

3. Expect similar response with tensorflow serving.