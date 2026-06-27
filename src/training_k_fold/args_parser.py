from training.args_parser import get_resnet_parser, add_args_resnet_parser

def get_kf_parser():
    parser = get_resnet_parser()
    parser.add_argument("--model_string", type=str, required=True)
    parser.add_argument("--folds", type=int, required=False, default=4)
    # parser.add_argument("--seed", type=int, required=False, default=4242)
    parser.add_argument("--fold_annotation_dir", type=str, required=False, default=None)
    return parser

def get_kf_vivit_parser():
    parser = get_kf_parser()
    parser.add_argument("--k", type=int, required=True, help="Fold number to train on (1-4)")
    return parser

def get_kf_args():
    parser = get_kf_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args

def get_kf_vivit_args():
    parser = get_kf_vivit_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args