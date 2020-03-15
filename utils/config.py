import argparse

__all__ = ['parse_cmd_args']


def create_parser():
    parser = argparse.ArgumentParser(description='Semi-supevised Training --PyTorch ')

    # Log and save
    parser.add_argument('--print-freq', default=20, type=int, metavar='N', help='display frequence (default: 20)')
    parser.add_argument('--save-freq', default=0, type=int, metavar='EPOCHS', help='checkpoint frequency(default: 0)')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, metavar='DIR')

    # Data
    parser.add_argument('--dataset', metavar='DATASET')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--num-labels', type=int, metavar='N', help='number of labeled samples')
    parser.add_argument('--sup-batch-size', default=100, type=int, metavar='N', help='batch size for supervised data (default: 100)')
    parser.add_argument('--usp-batch-size', default=100, type=int, metavar='N', help='batch size for unsupervised data (default: 100)')
    parser.add_argument('--data-root', type=str, metavar='DIR', default='./data-local')

    # Data pre-processing
    parser.add_argument('--data-twice', default=False, type=str2bool, metavar='BOOL', help='use two data stream (default: False)')
    parser.add_argument('--data-idxs', default=False, type=str2bool, metavar='BOOL', help='enable indexs of samples (default: False)')
    parser.add_argument('--label-exclude', type=str2bool, metavar='BOOL', help='exclude labeled samples in unsupervised batch')

    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH')
    parser.add_argument('--model', metavar='MODEL')
    parser.add_argument('--drop-ratio', default=0., type=float, help='ratio of dropout (default: 0)')

    # Optimization
    parser.add_argument('--epochs', type=int, metavar='N', help='number of total training epochs')
    parser.add_argument('--optim', default="sgd", type=str, metavar='TYPE', choices=['sgd', 'adam'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', default=False, type=str2bool, metavar='BOOL', help='use nesterov momentum (default: False)')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    
    # LR schecular
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='max learning rate (default: 0.1)')
    parser.add_argument('--lr-scheduler', default="cos", type=str, choices=['cos', 'multistep', 'exp-warmup', 'none'])
    parser.add_argument('--min-lr',  default=1e-4, type=float, metavar='LR', help='minimum learning rate (default: 1e-4)')
    parser.add_argument('--steps', type=int, nargs='+', metavar='N', help='decay steps for multistep scheduler')
    parser.add_argument('--gamma', type=float, help='factor of learning rate decay')
    parser.add_argument('--rampup-length', type=int, metavar='EPOCHS', help='length of the ramp-up')
    parser.add_argument('--rampdown-length', type=int, metavar='EPOCHS', help='length of the ramp-down')

    # Pseudo-Label 2013
    parser.add_argument('--t1', type=float, metavar='EPOCHS', help='T1')
    parser.add_argument('--t2', type=float, metavar='EPOCHS', help='T1')
    parser.add_argument('--soft', type=str2bool, help='use soft pseudo label')

    # VAT
    parser.add_argument('--xi', type=float, metavar='W', help='xi for VAT')
    parser.add_argument('--eps', type=float, metavar='W', help='epsilon for VAT')
    parser.add_argument('--n-power', type=int, metavar='N', help='the iteration number of power iteration method in VAT')
    
    # MeanTeacher-based method
    parser.add_argument('--ema-decay', type=float, metavar='W', help='ema weight decay')

    # Mixup-based method
    parser.add_argument('--mixup-alpha', type=float, metavar='W', help='mixup alpha for beta distribution')
 
    # Opt for loss
    parser.add_argument('--usp-weight', default=1.0, type=float, metavar='W', help='the upper of unsuperivsed weight (default: 1.0)')
    parser.add_argument('--weight-rampup', default=30, type=int, metavar='EPOCHS', help='the length of rampup weight (default: 30)')
    parser.add_argument('--ent-weight', type=float, metavar='W', help='the weight of minEnt regularization')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
