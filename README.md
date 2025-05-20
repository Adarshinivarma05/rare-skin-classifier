# Few-Shot Skin Condition Classification using ProtoPNet

Project structure ready.

# Rare Skin Condition Classifier using ProtoPNet

This project implements a **Few-Shot Classification** model for rare skin conditions using a Prototypical Network (ProtoPNet) architecture with PyTorch.

Skin diseases can be challenging to classify, especially rare conditions with limited data. This project leverages the **DermaMNIST** dataset from MedMNIST and a ProtoPNet model to improve explainability and accuracy in classifying skin conditions.

Features include:
- ProtoPNet architecture based on ResNet18 backbone.
- Few-shot learning with prototype-based classification.
- Data loading and preprocessing via the MedMNIST DermaMNIST dataset.
- Training pipeline with label smoothing, learning rate scheduler, early stopping, and TensorBoard logging.
- Explainability via Grad-CAM visualizations.
- Modular code structure with reusable utils for data loading, training, and visualization.

Project Structure:
```

data/
└── dermamnist/
└── temp/

models/
└── protopnet\_skin\_classifier.py

utils/
├── data\_loader.py
├── train.py
├── visualizer.py
└── explain\_utils.py

.gitignore
README.md
explain.py
test.py
train.py

````

Getting Started:

Prerequisites:
- Python 3.8+
- PyTorch
- torchvision
- medmnist
- matplotlib
- pytorch-grad-cam
- tensorboard

Install dependencies using:

```bash
pip install torch torchvision medmnist matplotlib pytorch-grad-cam tensorboard
````

Usage:

1. The dataset will automatically download using the `medmnist` library when running the training or evaluation scripts.

2. Run training with:

```bash
python train.py
```

3. Run the explanation script to generate Grad-CAM visualizations:

```bash
python explain.py
```

4. View training logs with:

```bash
tensorboard --logdir=runs
```

Author:
Spoorthi (spoorthichinthamalla)
Adarshini Varma (Adarshinivarma05)
Harsha Vardhan Reddy (h-rsh-19) 

License:
This project is licensed under the MIT License.
