# Model Analysis and Adversarial Input Generation

This project contains four main Jupyter notebooks for training, evaluating and analyzing neural network models:

## Setup
- Requires Python 3.8+
- Main dependencies: `PyTorch`, `NumPy`, `Matplotlib`
- Config file: `config.yaml` contains all configurable parameters

## Main Components
The notebooks must be run in the following order:

1. models_train.ipynb
- Trains the following models:
  - Custom dense network (on MNIST and Fashion-MNIST)
  - Custom conv network (on MNIST and Fashion-MNIST) 
  - MobileNetV3-Small (on CIFAR-10)
  - SqueezeNet (on CIFAR-10)
- Saves trained model parameters to `models/`

2. ` models_eval.ipynb `
- Evaluates all trained models on their respective datasets
- Determines the models' sizes (parameters)
- Stores evaluation results in `models/results.csv`

3. `models_get_sus.ipynb`
- Computes neuron suspiciousness values using Ochiai and Tarantula metrics
- Analyzes specific layers configured in config.yaml
- Stores results in `models-sus-values/` as pickle files
- Configuration specified in `suspiciousness-config.yaml`
  
4. ### `models_eval_adv_gen.ipynb`
- Performs adversarial input generation experiments
- Uses previously computed suspiciousness values to guide gradient-based input modifications
- Targets most suspicious neurons identified by Ochiai/Tarantula metrics
- Supports different model architectures and layer configurations
- Configuration specified in `train-config.yaml`
