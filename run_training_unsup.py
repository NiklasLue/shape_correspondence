import yaml
import os
import argparse

from project.train_dpfm import train_net_unsup


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='absolute path to files', default=os.path.dirname(os.path.realpath(__file__)), dest="path")
    parser.add_argument('-d', '--data_path', type=str, help='absolute path to data', default=os.path.dirname(os.path.realpath(__file__)), dest="data_path")
    parser.add_argument('-c', '--config', type=str, help='relative path to config file', default="dpfm/config/shrec16_cuts.yaml", dest="config_path")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"{args.path}/{args.config_path}", "r"))
    cfg["dataset"]["root_train"] = cfg["dataset"]["root_train"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["root_test"] = cfg["dataset"]["root_test"].replace("{{ABS_PATH}}", args.data_path)
    cfg["dataset"]["cache_dir"] = cfg["dataset"]["cache_dir"].replace("{{ABS_PATH}}", args.data_path)
    train_net_unsup(cfg)
