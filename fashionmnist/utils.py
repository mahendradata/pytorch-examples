import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def getFashionMNIST(datasets_dir: str = '../datasets', batch_size: int = 64) -> tuple:

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.2860,))  # Mean and Std of FashionMNIST dataset
    ])

    # Download training data from open datasets.
    train_data = datasets.FashionMNIST(
        root=datasets_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root=datasets_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # *** Preparing the DataLoader ***

    # Create data loaders.
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data, test_data, train_loader, test_loader



def train_once(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()  # Set model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # Move to specified device

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


def eval(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Set model to evaluation mode

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


def train(train_loader, test_loader, model, loss_fn, optimizer, epochs, device):
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_once(train_loader, model, loss_fn, optimizer, device)
        eval(test_loader, model, loss_fn, device)
    print('Done!')


def predict(model, x, y, device):
    model.eval()
    with torch.no_grad():
        x = x.to(device)  # Move to specified device
        x = torch.reshape(x, (-1, 1, 28, 28))
        pred = model(x)
        predicted, actual = pred[0].argmax(0), y
        print(f'Predicted: "{predicted}", Actual: "{actual}"')