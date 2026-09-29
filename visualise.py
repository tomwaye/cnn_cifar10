import torch
import matplotlib.pyplot as plt
from cnn_cifar10 import CNN, test_set, device, CIFAR_MEAN, CIFAR_STD

# Load the trained weights into a fresh model
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_cifar10.pt", map_location=device))
model.eval()

# Take 16 random test images
idx = torch.randint(len(test_set), (16,))
images = torch.stack([test_set[i][0] for i in idx])   # (16, 3, 32, 32)
labels = torch.tensor([test_set[i][1] for i in idx])

with torch.no_grad():
    probs = torch.softmax(model(images.to(device)), dim=1).cpu()
preds = probs.argmax(dim=1)

# Undo Normalize so the images display with their real colours
mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
std = torch.tensor(CIFAR_STD).view(3, 1, 1)
images = (images * std + mean).clamp(0, 1)

fig, axes = plt.subplots(4, 4, figsize=(8, 9))
for ax, img, p, t, pr in zip(axes.flat, images, preds, labels, probs):
    ax.imshow(img.permute(1, 2, 0))                    # (C, H, W) -> (H, W, C) for matplotlib
    correct = p == t
    mark = "✓" if correct else "✗"
    ax.set_title(f"{mark} {test_set.classes[p]} ({pr[p]:.0%})\ntrue: {test_set.classes[t]}",
                 fontsize=9, color="green" if correct else "red")
    ax.axis("off")
plt.tight_layout()
plt.show()

from torch.utils.data import DataLoader

loader = DataLoader(test_set, batch_size=512)
conf = torch.zeros(10, 10, dtype=torch.int64)
with torch.no_grad():
    for x, y in loader:
        p = model(x.to(device)).argmax(dim=1).cpu()
        for t, q in zip(y, p):
            conf[t, q] += 1

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(conf, cmap="Blues")                      # single-hue scale: darker = more images
ax.set_xticks(range(10), test_set.classes, rotation=45, ha="right")
ax.set_yticks(range(10), test_set.classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(10):
    for j in range(10):
        ax.text(j, i, conf[i, j].item(), ha="center", va="center", fontsize=8,
                color="white" if conf[i, j] > conf.max() / 2 else "black")
fig.colorbar(im)
plt.tight_layout()
plt.show()
