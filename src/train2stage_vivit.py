from training.args_parser import get_2stage_args
from training.train2stage_vivit import train2stage_vivit

def main():
    args = get_2stage_args()
    train2stage_vivit(**vars(args))

if __name__ == "__main__":
    main()

