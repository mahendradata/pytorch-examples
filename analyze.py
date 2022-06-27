from torchvision import datasets, transforms


def analyzing(dataset):

    train_set = dataset(
        root='./datasets',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    print('Size:', train_set.data.shape)
    print('Mean:', train_set.data.float().mean()/255)
    print('Std:', train_set.data.float().std()/255)


print('MNIST')
analyzing(datasets.MNIST)
print('FashionMNIST')
analyzing(datasets.FashionMNIST)