import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
from utils.utils import get_config
from dataloader import load_data, load_localization_data
from test_functions import detection_test, localization_test
from models.network import get_networks
import time
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    vgg, model = get_networks(config, load_checkpoint=True)
    last_checkpoint = config['last_checkpoint']

    # Detection test
    # else:
    _, test_dataloader1 = load_data(config)
    tic1 = time.time()
    with torch.no_grad():
        roc_auc1 = detection_test(model=model, model_teacher=vgg, test_dataloader=test_dataloader1, config=config)
    toc1 = time.time() - tic1
    print("detection time:", toc1)
    if roc_auc1 != 100:
        print("image RocAUC after {} epoch:".format(last_checkpoint), roc_auc1)

    # Localization test
    if config['localization_test']:
        train_dataloader, test_dataloader2, ground_truth = load_localization_data(config)
        roc_auc2 = localization_test(model=model, vgg=vgg, train_dataloader=train_dataloader, test_dataloader=test_dataloader2, ground_truth=ground_truth,
                                        config=config)
        print("pixel RocAUC after {} epoch:".format(last_checkpoint), roc_auc2)


if __name__ == '__main__':
    main()
