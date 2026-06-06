from training.train_medicalnet import train_medicalnet
from training.args_parser import get_resnet_args

def main():
    args = get_resnet_args()
    _ = train_medicalnet(**vars(args))


if __name__ == '__main__':
    main()
