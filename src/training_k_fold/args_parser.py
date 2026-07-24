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
    parser.add_argument("--time_stamp", type=str, required=False, default=None, help="Timestamp for the run")
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

def get_kf_2stage_parser():
    parser = get_kf_parser()
    parser.add_argument("--lr1", type=float, required=False, default=1e-4)
    parser.add_argument("--lr2", type=float, required=False, default=5e-5)
    parser.add_argument("--decay1", type=float, required=False, default=1e-3)
    parser.add_argument("--decay2", type=float, required=False, default=1e-3)
    parser.add_argument("--first_stage_epochs", type=int, required=False, default=20)
    parser.add_argument("--second_stage_epochs", type=int, required=False, default=50)
    return parser

def get_kf_2stage_args():
    parser = get_kf_2stage_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args

def get_kf_vivit_2stage_parser():
    parser = get_kf_2stage_parser()
    parser.add_argument("--k", type=int, required=True, help="Fold number to train on (1-4)")
    parser.add_argument("--time_stamp", type=str, required=False, default=None, help="Timestamp for the run")
    return parser

def get_kf_vivit_2stage_args():
    parser = get_kf_vivit_2stage_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args

def get_kf_luna16_double_parser():
    parser = get_resnet_parser()
    parser.add_argument("--model_string", type=str, required=True)
    parser.add_argument("--folds", type=int, required=False, default=4)
    parser.add_argument("--luna16_fold_annotation_dir", type=str, required=False, default=None)
    parser.add_argument("--luna16_train_image_dir", type=str, required=False, default=None)
    parser.add_argument("--luna16_validate_image_dir", type=str, required=False, default=None)
    parser.add_argument("--luna25_fold_annotation_dir", type=str, required=False, default=None)
    parser.add_argument("--luna25_train_image_dir", type=str, required=False, default=None)
    parser.add_argument("--luna25_validate_image_dir", type=str, required=False, default=None)
    parser.add_argument("--first_stage_epochs", type=int, required=False, default=30)
    parser.add_argument("--second_stage_epochs", type=int, required=False, default=30)
    parser.add_argument("--lr1", type=float, required=False, default=5e-5)
    parser.add_argument("--lr2", type=float, required=False, default=5e-5)
    parser.add_argument("--decay1", type=float, required=False, default=1e-3)
    parser.add_argument("--decay2", type=float, required=False, default=1e-3)
    return parser

def get_kf_luna16_double_args():
    parser = get_kf_luna16_double_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args

def get_kf_luna16_double_vivit_parser():
    parser = get_kf_luna16_double_parser()
    parser.add_argument("--k", type=int, required=True, help="Fold number to train on (1-4)")
    parser.add_argument("--time_stamp", type=str, required=False, default=None, help="Timestamp for the run")
    return parser

def get_kf_luna16_double_vivit_args():
    parser = get_kf_luna16_double_vivit_parser()
    args = parser.parse_args()
    add_args_resnet_parser(args)
    return args