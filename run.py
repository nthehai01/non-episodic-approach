"""Implementation of prototypical networks for Omniglot."""
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from torch.utils import tensorboard
import omniglot

from model.regular_model import RegularModel

NUM_TEST_TASKS = 600

SPLIT_PATH = os.environ['_SPLIT_PATH']
NUM_PRETRAIN_CLASSES = int(os.environ['_NUM_PRETRAIN_CLASSES'])
NUM_FINETUNE_CLASSES = int(os.environ['_NUM_FINETUNE_CLASSES'])
PRETRAIN_FOLDER_NAME = os.environ['_PRETRAIN_FOLDER_NAME']
FINETUNE_FOLDER_NAME = os.environ['_FINETUNE_FOLDER_NAME']


def main(args):
    phase = "finetune" if args.finetune else "pretrain"
    
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/{phase}/omniglot.num_classes_per_batch:{args.num_classes_per_batch}.num_images:{args.num_images}.lr:{args.learning_rate}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    
    num_outputs = NUM_FINETUNE_CLASSES if args.finetune else NUM_PRETRAIN_CLASSES
    net = RegularModel(num_outputs, args.learning_rate, log_dir)

    # Load checkpoint
    if args.checkpoint_step > -1:
        net.load(args.checkpoint_step, False)
    elif args.finetune:
        net.load(args.pretrained_weights, True)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on tasks with composition '
            f'num_classes_per_batch={args.num_classes_per_batch}, '
            f'num_images={args.num_images}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            phase,
            'train',
            args.batch_size,
            args.num_classes_per_batch,
            args.num_images,
            num_training_tasks
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            phase,
            'val',
            args.batch_size,
            args.num_classes_per_batch,
            args.num_images,
            args.batch_size * 4
        )
        net.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_classes_per_batch={args.num_classes_per_batch}, '
            f'num_images={args.num_images}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            phase,
            'test',
            1,
            args.num_classes_per_batch,
            args.num_images,
            NUM_TEST_TASKS
        )
        net.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--finetune', default=False, action='store_true',
                        help='pretrain or finetune')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_classes_per_batch', type=int, default=5,
                        help='number of classes in a batch')
    parser.add_argument('--num_images', type=int, default=1,
                        help='number of examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for the network (for ProtoNet-related only)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15001,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='pretrained weights to load for finetuning')

    main_args = parser.parse_args()
    main(main_args)
