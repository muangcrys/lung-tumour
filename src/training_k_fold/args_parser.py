from training.args_parser import get_resnet_parser, add_args_resnet_parser

def get_kf_parser():
    parser = get_resnet_parser()
    parser.add_arguement("--model_string", type=str, required=True)
    parser.add_arguement("--k", type=int, required=False, default=4)
    # parser.add_arguement("--seed", type=int, required=False, default=4242)
    parser.add_arguement("--fold_annotation_dir", type=str, required=False, default=None)
    return parser

def get_kf_args():
    parser = get_kf_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args