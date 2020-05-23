from ista_net import ISTA_Net
from argparse import ArgumentParser

parser = ArgumentParser(description='Train ISTA-Net-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--cs_ratio', type=int, default=4, help='from {1, 4, 10, 25, 40, 50}')

parser.add_argument('--matrix_path', type=str, required=True, help='path to sampling matrix')
parser.add_argument('--qinit_path', type=str, required=True, help='path to initialization matrix')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
cs_ratio = args.cs_ratio
phi_path = args.matrix_path
qinit_path = args.qinit_path
data_dir = args.data_dir
log_dir = args.log_dir


ista_net = ISTA_Net()

ista_net.load_phi(phi_path)
ista_net.load_qinit(qinit_path)


