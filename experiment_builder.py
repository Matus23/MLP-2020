import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from arg_extractor import get_args
args, device = get_args()
from torch.optim.adam import Adam

from storage_utils import save_statistics

class ExperimentBuilder(nn.Module):
    def __init__(self, model, ema_model, experiment_name, num_epochs, labeled_trainloader, unlabeled_trainloader ,
                                    val_loader,test_loader, use_gpu, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param model and ema_model: Two pytorch nn.Modules which implement network architectures.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param labeled_trainloader and unlabeled_trainloader: Two objects of the DataProvider type. Contains the training set.
        :param val_loader: An object of the DataProvider type. Contains the val set.
        :param test_loader: An object of the DataProvider type. Contains the test set.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = model
        self.ema_model = ema_model
        self.device = torch.cuda.current_device()

        # multiple gpu, single gpu, cpu
        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            self.ema_model.to(self.device)
            self.ema_model = nn.DataParallel(module=self.ema_model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu
            self.ema_model.to(self.device)
            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)

        # re-initialize network parameters
        self.labeled_trainloader = labeled_trainloader
        self.unlabeled_trainloader = unlabeled_trainloader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        self.train_criterion = SemiLoss()
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()


    def run_train_iter(self, inputs_x, targets_x, inputs_u, inputs_u2, epoch_idx, batch_idx):

        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1), 1)
        inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
        # inputs_u = inputs_u.to(self.device)
        # inputs_u2 = inputs_u2.to(self.device)
        #
        # with torch.no_grad():
        #     # compute guessed labels of unlabel samples
        #     outputs_u = self.model(inputs_u)
        #     outputs_u2 = self.model(inputs_u2)
        #     p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        #     pt = p**(1/args.T)
        #     targets_u = pt / pt.sum(dim=1, keepdim=True)
        #     targets_u = targets_u.detach()

        # mixup
        # all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        # all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        # l = np.random.beta(args.alpha, args.alpha)
        #
        # l = max(l, 1 - l)
        #
        # idx = torch.randperm(all_inputs.size(0))
        #
        # input_a, input_b = all_inputs, all_inputs[idx]
        # target_a, target_b = all_targets, all_targets[idx]
        #
        # mixed_input = l * input_a + (1 - l) * input_b
        # mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        # mixed_input = list(torch.split(mixed_input, batch_size))
        # mixed_input = interleave(mixed_input, batch_size)

        logits = self.model(inputs_x)
        # for input in mixed_input[1:]:
        #     logits.append(self.model(input))

        # put interleaved samples back
        # logits = interleave(logits, batch_size)
        # logits_x = logits[0]
        # logits_u = torch.cat(logits[1:], dim=0)

        Lx= self.train_criterion(logits, targets_x)

        loss = Lx

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema_optimizer.step()


        return loss.item(), 0, 0

    def run_evaluation_iter(self, inputs, targets):

        outputs = self.ema_model(inputs)  # forward the data in the model
        loss = self.criterion(outputs, targets)  # compute loss
        _, predicted = torch.max(outputs.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(targets.data).cpu()))  # compute accuracy
        return loss.item(), accuracy

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network_1'] = self.model.state_dict()  # save network parameter and other variables.
        state['network_2'] = self.ema_model.state_dict()
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def run_training_epoch(self, current_epoch_losses, epoch_idx):
        with tqdm.tqdm(total=args.val_iteration, file=sys.stdout) as pbar_train:  # create a progress bar for training
            labeled_train_iter = iter(self.labeled_trainloader)
            unlabeled_train_iter = iter(self.unlabeled_trainloader)
            self.model.train()
            for batch_idx in range(args.val_iteration):
                try:
                    inputs_x, targets_x = labeled_train_iter.next()
                except:
                    labeled_train_iter = iter(self.labeled_trainloader)
                    inputs_x, targets_x = labeled_train_iter.next()

                try:
                    (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(self.unlabeled_trainloader)
                    (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
            # get data batches
                loss, Lx, Lu = self.run_train_iter(inputs_x=inputs_x, targets_x=targets_x, inputs_u=inputs_u, inputs_u2=inputs_u2, epoch_idx=epoch_idx, batch_idx = batch_idx)  # take a training iter step
                current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                current_epoch_losses["train_loss_x"].append(Lx)  # add current iter acc to the train acc list
                current_epoch_losses["train_loss_u"].append(Lu)
                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}, Lx: {:.4f}, Lu: {:.4f}".format(loss, Lx, Lu))

        return current_epoch_losses

    def run_validation_epoch(self, current_epoch_losses, loader):
        if loader == "labelled":
            with tqdm.tqdm(total=len(self.labeled_trainloader), file=sys.stdout) as pbar_val:  # create a progress bar for validation
                self.ema_model.eval()
                for batch_idx, (inputs, targets) in enumerate(self.labeled_trainloader):  # get data batches
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    _, accuracy = self.run_evaluation_iter(inputs=inputs, targets=targets)  # run a validation iter
                    current_epoch_losses["train_acc"].append(accuracy)  # add current iter loss to val loss list.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("accuracy: {:.4f}".format(accuracy))

            return current_epoch_losses
        elif loader == "validate":
            with tqdm.tqdm(total=len(self.val_loader), file=sys.stdout) as pbar_val:  # create a progress bar for validation
                self.ema_model.eval()
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):  # get data batches
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    loss, accuracy = self.run_evaluation_iter(inputs=inputs, targets=targets)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

            return current_epoch_losses
        else:
            return current_epoch_losses

    def run_testing_epoch(self, current_epoch_losses):

        with tqdm.tqdm(total=len(self.test_loader), file=sys.stdout) as pbar_test:  # ini a progress bar
            self.ema_model.eval()
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):  # get data batches
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, accuracy = self.run_evaluation_iter(inputs=inputs, targets=targets)  # run a validation iter
                current_epoch_losses["test_loss"].append(loss)  # add current iter loss to val loss list.
                current_epoch_losses["test_acc"].append(accuracy)  # add current iter acc to val acc lst.
                pbar_test.update(1)  # add 1 step to the progress bar
                pbar_test.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

        return current_epoch_losses


    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.model.load_state_dict(state_dict=state['network_1'])
        self.ema_model.load_state_dict(state_dict=state['network_2'])
        return state['best_val_model_idx'], state['best_val_model_acc'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "train_loss_x": [],"train_loss_u": [],"val_acc": [],
                        "val_loss": [], "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_loss": [], "train_loss_x": [], "train_loss_u": [], "train_acc":[], "val_loss": [], "val_acc": []}

            # train with labelled and unlabelled training data
            current_epoch_losses = self.run_training_epoch(current_epoch_losses, epoch_idx)
            # report labelled training data accuracy
            current_epoch_losses = self.run_validation_epoch(current_epoch_losses, loader="labelled")
            # validate with validation data
            current_epoch_losses = self.run_validation_epoch(current_epoch_losses, loader="validate")

            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))
                # get mean of all metrics of current epoch metrics dict,
                # to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_acc'] = self.best_val_model_acc
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict

        current_epoch_losses = self.run_testing_epoch(current_epoch_losses=current_epoch_losses)

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format

        print("test loss: ", test_losses['test_loss'], ", test accuracy: ",test_losses['test_acc'] )

        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses

# a class to initialise ema_optimizer to optimize ema model
class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

# a function and a class to define SemiLoss
def linear_rampup(current, rampup_length=args.num_epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x):

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # Lu = torch.mean((probs_u - targets_u)**2)

        return Lx

# interleave function used during the training
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]