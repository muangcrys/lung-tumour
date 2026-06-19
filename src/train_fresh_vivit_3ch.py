from training.train_vivit import train_vivit
from training.args_parser import get_resnet_args

def main():
    args = get_resnet_args()
    _ = train_vivit(**vars(args), pretrained=False, num_channels=3)


if __name__ == '__main__':
    main()
