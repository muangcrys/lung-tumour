from training.train_fresh_medicalnet import train_fresh_medicalnet
from training.args_parser import get_resnet_args

def main():
    args = get_resnet_args()
    _ = train_fresh_medicalnet(**vars(args))


if __name__ == '__main__':
    main()
