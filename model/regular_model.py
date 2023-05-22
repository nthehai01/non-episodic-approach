"""Implementation of prototypical networks for Omniglot."""
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import util
from model import Backbone

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 10
NUM_TEST_TASKS = 600

NUM_CONV_LAYERS = 4


class RegularModelNetWork(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.parameters = Backbone.get_network(
            num_outputs=num_outputs,
            device=DEVICE
        )


    def forward(self, images):
        return Backbone.forward(images, self.parameters)


class RegularModel:
    def __init__(self, num_outputs, learning_rate, log_dir):
        self._network = RegularModelNetWork(num_outputs)
        self._optimizer = torch.optim.Adam(
            self._network.parameters.values(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0


    def _step(self, batch):
        loss_batch = []
        accuracy_batch = []
        for pair in batch:
            images, labels = pair
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Make predictions
            features = self._network(images)
            preds = F.log_softmax(features, dim=1)

            # Use F.cross_entropy to compute classification losses.
            loss = F.cross_entropy(preds, labels)
            loss_batch.append(loss)

            # Use util.score to compute accuracies.
            accuracy = util.score(preds, labels)
            accuracy_batch.append(accuracy)

        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_batch)
        )


    def train(self, dataloader_train, dataloader_val, writer):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            loss, accuracy = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'accuracy: {accuracy.item():.3f}, '
                )
                writer.add_scalar('loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy',
                    accuracy.item(),
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies = [], []
                    for val_task_batch in dataloader_val:
                        loss, accuracy = (
                            self._step(val_task_batch)
                        )
                        losses.append(loss.item())
                        accuracies.append(accuracy)
                    loss = np.mean(losses)
                    accuracy = np.mean(accuracies)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy:.3f}, '
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy',
                    accuracy,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)


    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[1])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )


    def load(self, checkpoint_step, is_finetune=False):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load
            is_finetune (bool): whether pretrained_checkpoint loaded for finetuning phase

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path, map_location=torch.device(DEVICE))

            trained_params = state['network_state_dict'].parameters

            if is_finetune:
                # remove linear head layer of pretrained network
                trained_params.pop(f'w{NUM_CONV_LAYERS}')
                trained_params.pop(f'b{NUM_CONV_LAYERS}')
            else:
                self._optimizer.load_state_dict(state['optimizer_state_dict'])
                self._start_train_step = checkpoint_step + 1
            
            # load pretrained network
            self._network.parameters.update(trained_params)

            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )


    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network,
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')
