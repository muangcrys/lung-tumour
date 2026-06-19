from training.args_parser import get_2stage_args
from training.train2stage_resnet3d import train2stage_resnet3d

def main():
    args = get_2stage_args()
    train2stage_resnet3d(**vars(args))

if __name__ == "__main__":
    main()

