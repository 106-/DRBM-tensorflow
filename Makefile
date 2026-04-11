.PHONY: sync format train-mnist train-cifar train-fashion-mnist train-generative train-olivetti train-urban clean

EPOCHS ?= 100
CONFIG ?= config/mnist/h100/continuous.json

sync:
	uv sync

format:
	uvx ruff format .

train-mnist:
	uv run python train_mnist.py $(CONFIG) $(EPOCHS)

train-cifar:
	uv run python train_cifar.py $(CONFIG) $(EPOCHS)

train-fashion-mnist:
	uv run python train_fashion_mnist.py $(CONFIG) $(EPOCHS)

train-generative:
	uv run python train_generative.py $(CONFIG) $(EPOCHS)

train-olivetti:
	uv run python train_olivetti.py $(CONFIG) $(EPOCHS)

train-urban:
	uv run python train_urban.py $(CONFIG) $(EPOCHS)

clean:
	rm -rf .venv
