from training.args_parser import get_2stage_args
from training.train2stage_medicalnet import train2stage_medicalnet

def main():
    args = get_2stage_args()
    train2stage_medicalnet(**vars(args))

if __name__ == "__main__":
    main()

