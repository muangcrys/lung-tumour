import argparse
from pathlib import Path


def get_resnet_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=18,
                        help='depth of resnet (default: 18)', required=False)
    parser.add_argument('--ckt_path', type=str, default=None,
                        help='path to checkpoint', required=False)
    parser.add_argument('--ckt_num_classes', type=int, default=None,
                        help='number of classes AS IN CHECKPOINT FILE', required=False)
    parser.add_argument('--keep_classifier', action='store_true',
                        help='whether to keep the pretrained classifier layer')
    parser.add_argument('--train_classifier_only', action='store_true',
                        help='whether to train classifier only (freeze all other layers)')
    parser.add_argument('--train_annotation', type=str, default=None, required=False,
                        help='path to training annotation')
    parser.add_argument('--train_image_dir', type=str, default=None, required=False,
                        help='path to training images directory')
    parser.add_argument('--val_annotation', type=str, default=None, required=False,
                        help='path to validation annotation')
    parser.add_argument('--val_image_dir', type=str, default=None, required=False,
                        help='path to validation images directory')
    parser.add_argument('--epochs', type=int, default=50, required=False,
                        help='number of epochs (default: 50)')
    parser.add_argument('--optim_type', type=str, default='AdamW', required=False,
                        help='type of optimizer (default: AdamW)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, required=False,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--pos_weight', type=float, default=2.0, required=False,
                        help='weight for positive class for weighted cross entropy (default: 2.0)')
    parser.add_argument('--batch_size', type=int, default=16, required=False,
                        help='batch size (default: 16)')
    parser.add_argument('--num_workers', type=int, default=0, required=False,
                        help='number of workers (default: 0)')
    parser.add_argument('--seed', type=int, default=4242, required=False,
                        help='random seed (default: 4242)')
    parser.add_argument('--report_frequency', type=int, default=10, required=False,
                        help='report frequency (default: 10)')
    parser.add_argument('--base_directory', type=str, default=None, required=False,
                        help='base directory (default: None)')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='device (default: None) (agnostic)')

    return parser


def add_args_resnet_parser(args: argparse.Namespace):
    args.replace_classifier = not args.keep_classifier


def get_resnet_args():
    parser = get_resnet_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args


def get_2stage_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=18,
                        help='depth of resnet (default: 18)', required=False)
    parser.add_argument('--ckt_path', type=str, default=None,
                        help='path to checkpoint', required=False)
    parser.add_argument('--ckt_num_classes', type=int, default=None,
                        help='number of classes AS IN CHECKPOINT FILE', required=False)
    parser.add_argument('--keep_classifier', action='store_true',
                        help='whether to keep the pretrained classifier layer')
    parser.add_argument('--train_annotation', type=str, default=None, required=False,
                        help='path to training annotation')
    parser.add_argument('--train_image_dir', type=str, default=None, required=False,
                        help='path to training images directory')
    parser.add_argument('--val_annotation', type=str, default=None, required=False,
                        help='path to validation annotation')
    parser.add_argument('--val_image_dir', type=str, default=None, required=False,
                        help='path to validation images directory')
    parser.add_argument('--first_stage_epochs', type=int, default=20, required=False,
                        help='number of epochs for classifier only training (default: 20)')
    parser.add_argument('--second_stage_epochs', type=int, default=50, required=False,
                        help='number of epochs for overall fine tuning (default: 50)')
    parser.add_argument('--lr1', type=float, default=1e-4, required=False,)
    parser.add_argument('--lr2', type=float, default=5e-5, required=False,)
    parser.add_argument('--decay1', type=float, default=1e-4, required=False,)
    parser.add_argument('--decay2', type=float, default=1e-4, required=False,)
    parser.add_argument('--bce_weight1', type=float, default=2.0, required=False,)
    parser.add_argument('--bce_weight2', type=float, default=2.0, required=False,)
    parser.add_argument('--metric', type=str, default='auroc', required=False,)
    parser.add_argument('--higher_is_better', action='store_true', default=None)
    parser.add_argument('--lower_is_better', action='store_false', dest='higher_is_better', default=None)
    parser.add_argument('--threshold', type=float, default=0.5, required=False,)
    parser.add_argument('--batch_size', type=int, default=16, required=False,
                        help='batch size (default: 16)')
    parser.add_argument('--num_workers', type=int, default=0, required=False,
                        help='number of workers (default: 0)')
    parser.add_argument('--seed', type=int, default=4242, required=False,
                        help='random seed (default: 4242)')
    parser.add_argument('--report_frequency', type=int, default=10, required=False,
                        help='report frequency (default: 10)')
    parser.add_argument('--base_directory', type=str, default=None, required=False,
                        help='base directory (default: None)')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='device (default: None) (agnostic)')

    return parser

def get_2stage_args():
    parser = get_2stage_parser()
    args = parser.parse_args()
    return args