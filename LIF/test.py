import snntorch as snn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Global params
batch_size = 128
data_path = '../data/mnist'
num_classes = 10

# prepare for the dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])  # 转为tensor，并归一化至[0-1]

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
print(f"The size of mnist_train is {len(mnist_train)}")
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# generate spike information
num_steps = 10
raw_vector = torch.ones(num_steps) * 0.5
