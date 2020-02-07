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
np.random.seed(args.seed)



# Fetch dataset (MixMatch)
transform_train = transforms.Compose([
    data_providers.RandomPadandCrop(32),
    data_providers.RandomFlip(),
    data_providers.ToTensor(),
])

transform_val = transforms.Compose([
    data_providers.ToTensor(),
])

dataset_name = args.dataset_name
num_channels = 0
if dataset_name == "cifar10":
    train_labeled_set, train_unlabeled_set, val_set, test_set = data_providers.get_cifar10('./data', args.n_labeled, transform_train=transform_train, transform_val=transform_val)
    num_channels = 3
elif dataset_name == "mnist":
    train_labeled_set, train_unlabeled_set, val_set, test_set = data_providers.get_MNIST('./data', args.n_labeled,
                                                                                           transform_train=transform_train,
                                                                                           transform_val=transform_val)
    num_channels = 1
labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Create two models (MixMatch)
def create_model(ema=False):
    model = WideResNet(num_classes=10, num_channels=num_channels)


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
