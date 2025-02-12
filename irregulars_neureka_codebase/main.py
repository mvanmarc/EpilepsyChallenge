
from utils.config import process_config, setup_logger, process_config_default
import argparse
import shutil
shutil._USE_CP_SENDFILE = False
from train import train

def main(config_path, default_config_path, args):
    setup_logger()

    config = process_config_default(config_path, default_config_path)

    m = ""
    enc_m = ""
    if "lr" in args and args.lr is not None:
        config.optimizer.learning_rate = float(args.lr)
        m += "_lr{}".format(args.lr)
        enc_m += "_lr{}".format(args.lr)
    if "wd" in args and args.wd is not None:
        config.optimizer.weight_decay = float(args.wd)
        m += "_wd{}".format(args.wd)
        enc_m += "_wd{}".format(args.wd)
    if "mm" in args and args.mm is not None:
        config.optimizer.momentum = float(args.mm)
        m += "_mm{}".format(args.mm)
        enc_m += "_mm{}".format(args.mm)
    if "rsz" in args and args.rsz is not None:
        config.dataset.ratio_sz_nsz = float(args.rsz)
        m += "_rsz{}".format(args.rsz)
        enc_m += "_rsz{}".format(args.rsz)

    config.model.save_dir = config.model.save_dir.format(m)

    # if enc_m != "":
    if hasattr(config.model, "encoders"):
        for i in range(len(config.model.encoders)):
            config.model.encoders[i].pretrainedEncoder.dir = config.model.encoders[i].pretrainedEncoder.dir.format(enc_m)

    train(config)


parser = argparse.ArgumentParser(description="My Command Line Program")
parser.add_argument('--config', help="Number of config file")
parser.add_argument('--default_config', help="Number of config file")
parser.add_argument('--lr', required=False, help="Learning Rate", default=None)
parser.add_argument('--wd', required=False, help="Weight Decay", default=None)
parser.add_argument('--mm', required=False, help="Optimizer Momentum", default=None)
parser.add_argument('--rsz', required=False, help="ratio_sz_nsz", default=None)
args = parser.parse_args()

for var_name in vars(args):
    var_value = getattr(args, var_name)
    if var_value == "None":
        setattr(args, var_name, None)

print(args)


main(config_path=args.config, default_config_path=args.default_config, args=args)