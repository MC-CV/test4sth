import warnings
warnings.filterwarnings("ignore")
from test import *
from utils.utils import *
from dataloader import *
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test
from loss_functions import *
import argparse
import torch
from models.network import *
from collections import OrderedDict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")


def train(config):
    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']
    last_checkpoint = config['last_checkpoint']
    network = config['network']

    checkpoint_path = "./{}/{}/checkpoints/".format(config['experiment_name'], config['mydata_name'])

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = load_data(config)
    # if continue_train:
    #     vgg, model = get_networks(config, load_checkpoint=True)
    model_teacher, model = get_networks(config)


    # Criteria And Optimizers
    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)
    if network == "resnet18":
        criterion = MseDirectionLoss_resnet(lamda)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # if continue_train:
    #     optimizer.load_state_dict(
    #         torch.load('{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))

    losses = []
    roc_aucs = []
    # if continue_train:
    #     with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, last_checkpoint), 'rb') as f:
    #         roc_aucs = pickle.load(f)

    for epoch in range(num_epochs + 1):
        model.cuda()
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            X = data[0]
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()

            if network == "vgg":
                output_pred = model.forward(X)
                output_real = model_teacher(X)
            elif network == "resnet18":
                output_pred = model.forward(X)
                model_teacher.cuda()
                outputs = []

                def hook(module, input, output):
                    outputs.append(output)
                hook1 = model_teacher.layer1[-1].register_forward_hook(hook)
                hook2 = model_teacher.layer2[-1].register_forward_hook(hook)
                hook3 = model_teacher.layer3[-1].register_forward_hook(hook)
                hook4 = model_teacher.layer4[-1].register_forward_hook(hook)
                train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('layer4', [])])
                with torch.no_grad():
                    model_teacher(X)
                hook1.remove()
                hook2.remove()
                hook3.remove()
                hook4.remove()
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.detach())
                # initialize hook outputs
                outputs = []
                output_real = []
                output_real.append(train_outputs['layer1'])
                output_real.append(train_outputs['layer2'])
                output_real.append(train_outputs['layer3'])
                output_real.append(train_outputs['layer4'])

            total_loss = criterion(output_pred, output_real)

            # Add loss to the list
            epoch_loss += total_loss.item()
            losses.append(total_loss.item())

            # Clear the previous gradients
            optimizer.zero_grad()
            # Compute gradients
            total_loss.backward()
            # Adjust weights
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        # if epoch % 10 == 0:
        roc_auc = detection_test(model, model_teacher, test_dataloader, config)
        roc_aucs.append(roc_auc)
        print("RocAUC at epoch {}:".format(epoch), roc_auc)

        roc_max = max(roc_aucs)
        print("roc_max:", roc_max)
        if roc_max == roc_auc and roc_max > 0.8:
            print("max_epoch:{}".format(epoch))
            torch.save(model.state_dict(),
                           '{}{}_epoch_{}.pth'.format(checkpoint_path, network, epoch))

        # if epoch == 10:
        #     torch.save(model.state_dict(),
        #                '{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
        
            # torch.save(optimizer.state_dict(),
            #            '{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            # with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, epoch),
            #           'wb') as f:
            #     pickle.dump(roc_aucs, f)


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
