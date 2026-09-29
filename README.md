# CNN on CIFAR-10

A convolutional neural network in PyTorch that classifies CIFAR-10 images into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

It follows on from a fully connected network I built earlier (first in NumPy, then in PyTorch), which reached 97.7% on MNIST but only 49.9% on CIFAR-10. The goal here was to beat that 49.9% with a CNN.

**Result: about 83% test accuracy**, with around 357k parameters and a few minutes of training on an Apple Silicon GPU.

## Results

Each row adds one change to the row above it:

| Model | Test accuracy |
|---|---|
| Fully connected network (previous project) | 49.9% |
| CNN: 2 conv blocks, SGD with momentum, 5 epochs | 72.6% |
| + data augmentation (random crop + horizontal flip), 10 epochs | 74.9% |
| + batch normalisation | 75.8% |
| + third conv block, learning rate 0.05 | 76.2% |
| + cosine annealing learning-rate schedule | 79.9% |
| + 20 epochs instead of 10 | **83.0%** |

What each step showed:

- **Convolutions** keep the image's spatial structure and share weights across positions. That alone gained over 20 points on the fully connected network, which had to flatten each image into 3072 unrelated numbers.
- **Data augmentation** stopped the model memorising the training set. Without it, training accuracy ran 8.5 points ahead of test accuracy after 5 epochs. With it, test accuracy stayed at or above training accuracy.
- **Batch normalisation** made only a small difference in a 2-layer network. Raising the learning rate to 0.05 without a schedule made results *worse* (69%), because the large unnormalised Linear layer was unstable.
- **A third conv block** halved the size of that Linear layer and let the model train well at learning rate 0.05, while using a third fewer parameters.
- **The cosine schedule** made the final epoch the best one. Before, test accuracy moved around by 1–2 points at the end of training because the learning rate stayed high.

I also tried a wider VGG-style model (two convs per block, 64/128/256 channels, 1.7M parameters). It was about 2.5 minutes per epoch, which wasn't worth the likely gain for this project, so I kept the smaller model.

## Architecture

```
Input                                   (3, 32, 32)
Conv 3→32,   3×3  → BatchNorm → ReLU → MaxPool 2×2   (32, 16, 16)
Conv 32→64,  3×3  → BatchNorm → ReLU → MaxPool 2×2   (64, 8, 8)
Conv 64→128, 3×3  → BatchNorm → ReLU → MaxPool 2×2   (128, 4, 4)
Flatten                                               2048
Linear 2048→128 → ReLU
Linear 128→10                                         logits
```

## Training setup

- **Data:** CIFAR-10, 50,000 training and 10,000 test images, normalised with the per-channel mean and standard deviation of the training set.
- **Augmentation (training only):** random horizontal flip, and random 32×32 crop from the image padded by 4 pixels.
- **Loss:** cross-entropy.
- **Optimiser:** SGD, learning rate 0.05, momentum 0.9.
- **Schedule:** cosine annealing over 20 epochs.

## Running it

Requires Python 3 with PyTorch and torchvision.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
python cnn_cifar10.py
```

CIFAR-10 downloads to `./data` on the first run. The script uses Apple's GPU (MPS) if it's available, and the CPU otherwise. It prints training and test loss and accuracy for each epoch, then eight sample predictions from the test set, and saves the trained weights to `cnn_cifar10.pt`.

## Using the saved weights

`cnn_cifar10.pt` holds the model's `state_dict`. To load it, copy the `CNN` class definition from `cnn_cifar10.py`. Importing the script would start training, because the training code runs at the top level of the file.

```python
model = CNN()
model.load_state_dict(torch.load("cnn_cifar10.pt"))
model.eval()
```
