.PHONY: sync format train-mnist train-cifar train-fashion-mnist train-generative train-olivetti train-urban clean

EPOCHS ?= 100

MNIST_CONFIG       ?= config/mnist/h100/continuous.json
CIFAR_CONFIG       ?= config/cifar/h500/continuous_sparse.json
FASHION_CONFIG     ?= config/mnist/h100/continuous.json
GENERATIVE_CONFIG  ?= config/generative/h100/continuous.json
OLIVETTI_CONFIG    ?= config/olivetti/continuous.json
URBAN_CONFIG       ?= config/urban/continuous.json

sync:
	uv sync

format:
	uvx ruff format .

train-mnist:
	uv run python train_mnist.py $(MNIST_CONFIG) $(EPOCHS)

train-cifar:
	uv run python train_cifar.py $(CIFAR_CONFIG) $(EPOCHS)

train-fashion-mnist:
	uv run python train_fashion_mnist.py $(FASHION_CONFIG) $(EPOCHS)

train-generative:
	uv run python train_generative.py $(GENERATIVE_CONFIG) $(EPOCHS)

train-olivetti:
	uv run python train_olivetti.py $(OLIVETTI_CONFIG) $(EPOCHS)

train-urban:
	uv run python train_urban.py $(URBAN_CONFIG) $(EPOCHS)

clean:
	rm -rf .venv
