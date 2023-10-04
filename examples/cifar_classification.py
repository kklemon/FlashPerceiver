import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from pytorch_lamb import Lamb
from tqdm import tqdm
from fast_perceiver import utils, Perceiver
from fast_perceiver.adapters import ImageAdapter
from fast_perceiver.utils.encodings import NeRFPositionalEncoding
from fast_perceiver.utils.training import CosineWithWarmupLR


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--device', default='cuda')
parser.add_argument('--data_root', default='./data')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

rng = torch.Generator(device=args.device)

train_set = CIFAR10(args.data_root, train=True, download=True,transform=transform_train)
test_set = CIFAR10(args.data_root, train=False, download=True,transform=transform_test)

valid_set, train_set = random_split(train_set, [5000, 45000])

train_batches = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
valid_batches = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
test_batches = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

print('==> Building model..')

model = nn.Sequential(
    ImageAdapter(
        embed_dim=64,
        pos_encoding=NeRFPositionalEncoding(2),
    ),
    Perceiver(
        input_dim=64,
        depth=1,
        output_dim=10,
        num_latents=128,
        latent_dim=256,
        cross_attn_dropout=0.2,
        latent_attn_dropout=0.2,
        self_per_cross_attn=4
    )
).to(args.device)

criterion = nn.CrossEntropyLoss()
optimizer = Lamb(
    model.parameters(),
    lr=args.lr,
    weight_decay=1e-4
)
scheduler = CosineWithWarmupLR(
    optimizer,
    training_steps=args.epochs * len(train_batches),
    warmup_steps=1000
)

print(f'Number of parameters: {utils.numel(model):,}')

def train(dataset, log_prefix):
    model.train()

    with tqdm(dataset) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            optimizer.zero_grad()

            with torch.autocast(args.device):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()

            optimizer.step()
            scheduler.step()

            acc = outputs.argmax(-1).eq(targets).float().mean().item()
            lr = scheduler.get_last_lr()[0]

            pbar.set_description(
                f'{log_prefix} | loss: {loss.item():.3f}, acc: {100.0 * acc:.3f}, lr: {lr:.3e}'
            )


@torch.no_grad()
def evaluate(dataset, log_prefix='VALID'):
    model.eval()

    loss = 0
    correct = 0
    total = 0

    with torch.autocast(args.device):
        for inputs, targets in dataset:
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs = model(inputs)

            loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += outputs.argmax(-1).eq(targets).sum().item()

        print(f'{log_prefix} | loss: {loss / total:.3f}, acc: {100.0 * correct / total:.3f}')


for epoch in range(args.epochs):
    train(train_batches, f'TRAIN | EPOCH {epoch}')
    evaluate(valid_batches, f'VALID | EPOCH {epoch}')

evaluate(test_batches, f'TEST ')