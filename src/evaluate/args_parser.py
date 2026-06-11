import argparse

def get_evaluate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_directory', required=True, type=str)
    parser.add_argument('--annotation', default=None, type=str, required=False)
    parser.add_argument('--image_dir', default=None, type=str, required=False)
    parser.add_argument('--preprocessing', default=None, type=str, required=False)
    parser.add_argument('--metrics-directory', default=None, type=str, required=False)
    parser.add_argument('--skip_plot_loss', action='store_true', required=False)
    parser.add_argument('--skip_final_model', action='store_true', required=False)
    parser.add_argument('--skip_best_model', action='store_true', required=False)
    parser.add_argument('--threshold', default=0.5, type=float, required=False)
    parser.add_argument('--model_type', default=None, type=str, required=False)
    parser.add_argument('--depth', default=None, type=int, required=False)
    parser.add_argument('--channels', default=None, type=int, required=False)
    parser.add_argument('--batch_size', default=8, type=int, required=False)
    parser.add_argument('--num_workers', default=0, type=int, required=False)
    parser.add_argument('--device', default=None, type=str, required=False)
    return parser

def add_args_evaluate_parser(args: argparse.Namespace):
    args.plot_loss = not args.skip_plot_loss
    args.final_model = not args.skip_final_model
    args.best_model = not args.skip_best_model

def get_evaluate_args():
    parser = get_evaluate_parser()
    args = parser.parse_args()
    add_args_evaluate_parser(args)
    return args