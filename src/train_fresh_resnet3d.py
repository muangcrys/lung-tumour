from training.train_fresh_resnet3d import train_fresh_resnet3d
from training.args_parser import get_resnet_args

def main():
    args = get_resnet_args()
    _ = train_fresh_resnet3d(**vars(args))


if __name__ == '__main__':
    main()
