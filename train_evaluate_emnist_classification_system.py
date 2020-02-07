import numpy as np

import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from model_architectures import WideResNet
import torch.utils.data as data

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

from torchvision import transforms
import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed



if args.dataset_name == 'emnist':
    train_data = data_providers.EMNISTDataProvider('train', batch_size=args.batch_size,
                                                   rng=rng,
                                                   flatten=False)  # initialize our rngs using the argument set seed
    val_data = data_providers.EMNISTDataProvider('valid', batch_size=args.batch_size,
                                                 rng=rng,
                                                 flatten=False)  # initialize our rngs using the argument set seed
    test_data = data_providers.EMNISTDataProvider('test', batch_size=args.batch_size,
                                                  rng=rng,
                                                  flatten=False)  # initialize our rngs using the argument set seed
    num_output_classes = train_data.num_classes

elif args.dataset_name == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(n_holes=1, length=14),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = data_providers.CIFAR10(root='data', set_name='train', download=True, transform=transform_train)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

    valset = data_providers.CIFAR10(root='data', set_name='val', download=True, transform=transform_test)
    val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)

    testset = data_providers.CIFAR10(root='data', set_name='test', download=True, transform=transform_test)
    test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_output_classes = 10

elif args.dataset_name == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(n_holes=1, length=14),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = data_providers.CIFAR100(root='data', set_name='train', download=True, transform=transform_train)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

    valset = data_providers.CIFAR100(root='data', set_name='val', download=True, transform=transform_test)
    val_data = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=4)

    testset = data_providers.CIFAR100(root='data', set_name='test', download=True, transform=transform_test)
    test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    num_output_classes = 100




# Fetch dataset (MixMatch)
transform_train = transforms.Compose([
    data_providers.RandomPadandCrop(32),
    data_providers.RandomFlip(),
    data_providers.ToTensor(),
])

transform_val = transforms.Compose([
    data_providers.ToTensor(),
])

train_labeled_set, train_unlabeled_set, val_set, test_set = data_providers.get_cifar10('./data', args.n_labeled, transform_train=transform_train, transform_val=transform_val)
labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Create two models (MixMatch)
def create_model(ema=False):
    model = WideResNet(num_classes=10)


    if ema:
        for param in model.parameters():
            param.detach_()

    return model
model = create_model()
ema_model = create_model(ema=True)




conv_experiment = ExperimentBuilder(model=model, ema_model = ema_model, use_gpu=args.use_gpu,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    continue_from_epoch=args.continue_from_epoch,
                                    labeled_trainloader=labeled_trainloader, unlabeled_trainloader = unlabeled_trainloader,
                                    val_loader=val_loader,
                                    test_loader=test_loader)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
