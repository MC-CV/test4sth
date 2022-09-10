import torch
from torch import nn
from torchvision.models import vgg16
from pathlib import Path
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, resnet18


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        # placeholder for the gradients
        self.gradients = None
        self.activation = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, target_layer=11):
        result = []
        for i in range(len((self.features))):
            x = self.features[i](x)
            # if i == target_layer:
            #     self.activation = x
            #     h = x.register_hook(self.activations_hook)
            if i == 2 or i == 5 or i == 8 or i == 11 or i == 14 or i == 17 or i == 20 or i == 23 or i == 26 or i == 29 or i == 32 or i == 35 or i == 38:
                result.append(x)

        return result

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation


def make_layers(cfg, use_bias, batch_norm=False):
    layers = []
    in_channels = 3
    outputs = []
    for i in range(len(cfg)):
        if cfg[i] == 'O':
            outputs.append(nn.Sequential(*layers))
        elif cfg[i] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1, bias=use_bias)
            torch.nn.init.xavier_uniform_(conv2d.weight)
            if batch_norm and cfg[i + 1] != 'M':
                layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg[i]
    return nn.Sequential(*layers)


def make_arch(idx, cfg, use_bias, batch_norm=False):
    return VGG(make_layers(cfg[idx], use_bias, batch_norm=batch_norm))


class Vgg16(torch.nn.Module):
    def __init__(self, pretrain):
        super(Vgg16, self).__init__()
        features = list(vgg16('vgg16-397923af.pth').features)

        # if not pretrain:
        #     for ind, f in enumerate(features):
        #         # nn.init.xavier_normal_(f)
        #         if type(f) is torch.nn.modules.conv.Conv2d:
        #             torch.nn.init.xavier_uniform(f.weight)
        #             print("Initialized", ind, f)
        #         else:
        #             print("Bypassed", ind, f)
        #     # print("Pre-trained Network loaded")
        self.features = nn.ModuleList(features).eval()
        self.output = []

    def forward(self, x):
        output = []
        for i in range(31):
            x = self.features[i](x)
            if i == 1 or i == 4 or i == 6 or i == 9 or i == 11 or i == 13 or i == 16 or i == 18 or i == 20 or i == 23 or i == 25 or i == 27 or i == 30:
                output.append(x)
        return output


# define structure
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MYResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(MYResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)               # the first layer

        self.layer1 = self._make_layer(block, 64, 2, stride=1)             # four layers 2-5
        self.layer2 = self._make_layer(block, 128, 2, stride=2)            # four layers 6-9
        self.layer3 = self._make_layer(block, 256, 2, stride=2)            # four layers 10-13
        self.layer4 = self._make_layer(block, 512, 2, stride=2)            # four layers 14-17

        self.fc = nn.Linear(512, num_classes)                                         # the last layer

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, 16, stride))
                self.in_planes = 16
            else:
                layers.append(block(16, planes, 1))
                self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        output.append(x)
        x = self.layer2(x)
        output.append(x)
        x = self.layer3(x)
        output.append(x)
        x = self.layer4(x)
        output.append(x)
        # x = F.avg_pool2d(x, 4)
        # x = x.view(x.size(0), -1)
        # out = self.fc(x)
        return output


def get_networks(config, load_checkpoint=False):
    if config['network'] == "vgg":
        equal_network_size = config['equal_network_size']
        pretrain = config['pretrain']
        experiment_name = config['experiment_name']
        dataset_name = config['dataset_name']
        normal_class = config['normal_class']
        use_bias = config['use_bias']
        mydata_name = config['mydata_name']

        cfg = {
            'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'B': [16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16, 16, 512, 'M'],
        }

        if equal_network_size:
            config_type = 'A'
        else:
            config_type = 'B'

        vgg = Vgg16(pretrain).cuda()
        model = make_arch(config_type, cfg, use_bias, True).cuda()

        # print layers
        # for j, item in enumerate((vgg.features)):
        #     print('layer : {} {}'.format(j, item))
        # for j, item in enumerate((model.features)):
        #     print('layer : {} {}'.format(j, item))

        if load_checkpoint:
            last_checkpoint = config['last_checkpoint']
            checkpoint_path = "./{}/{}/checkpoints/".format(experiment_name, mydata_name)
            # checkpoint_path = "../input/mybishe/mypth/"
            # checkpoint_path = "./mypth/"
            model.load_state_dict(
                torch.load('{}{}_epoch_{}.pth'.format(checkpoint_path, config['network'], last_checkpoint)))
            if not pretrain:
                vgg.load_state_dict(
                    torch.load('{}Source_{}_random_vgg.pth'.format(checkpoint_path, normal_class)))
        elif not pretrain:
            # checkpoint_path = "./{}/{}/checkpoints/".format(experiment_name, dataset_name)
            checkpoint_path = "./checkpoints/"
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

            torch.save(vgg.state_dict(), '{}Source_{}_random_vgg.pth'.format(checkpoint_path, normal_class))
            print("Source Checkpoint saved!")

        return vgg, model
    elif config['network'] == "resnet18":
        experiment_name = config['experiment_name']
        mydata_name = config['mydata_name']

        resnet = resnet18(pretrained=True, progress=True).cuda()
        model = MYResNet(BasicBlock)

        if load_checkpoint:
            last_checkpoint = config['last_checkpoint']
            checkpoint_path = "./{}/{}/checkpoints/".format(experiment_name, mydata_name)
            model.load_state_dict(
                torch.load('{}{}_epoch_{}.pth'.format(checkpoint_path, config['network'], last_checkpoint)))

        return resnet, model