from torch import nn
from sklearn.metrics import roc_curve, auc
import imp
# utils=imp.load_source('utils','../input/mybishe/data/utils/utils.py')
# from utils import morphological_process, convert_to_grayscale, max_regarding_to_abs
from utils.utils import morphological_process, convert_to_grayscale, max_regarding_to_abs
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch
from torch.autograd import Variable
from copy import deepcopy
from torch.nn import ReLU
import os
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from scipy.spatial.distance import mahalanobis
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
from collections import OrderedDict



def detection_test(model, model_teacher, test_dataloader, config):
    normal_class = config["normal_class"]
    lamda = config['lamda']
    dataset_name = config['dataset_name']
    direction_only = config['direction_loss_only']
    module = config['module']
    thre = config['thre']

    if dataset_name != "mvtec":
        target_class = normal_class
    else:
        mvtec_good_dict = {'bottle': 3, 'cable': 5, 'capsule': 2, 'carpet': 2,
                           'grid': 3, 'hazelnut': 2, 'leather': 4, 'metal_nut': 3, 'pill': 5,
                           'screw': 0, 'tile': 2, 'toothbrush': 1, 'transistor': 3, 'wood': 2,
                           'zipper': 4
                           }
        target_class = mvtec_good_dict[normal_class]

    similarity_loss = torch.nn.CosineSimilarity()
    label_score = []
    model.eval()
    model.cuda()
    tic = time.time()
    for data in test_dataloader:
        X, Y = data
        if X.shape[1] == 1:
            X = X.repeat(1, 3, 1, 1)
        X = Variable(X).cuda()
        if config['network'] == "vgg":
            output_pred = model.forward(X)
            output_real = model_teacher(X)
            y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[3], output_pred[6], output_pred[9], output_pred[12]
            y_0, y_1, y_2, y_3 = output_real[3], output_real[6], output_real[9], output_real[12]
        elif config['network'] == 'resnet18':
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
            y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[0], output_pred[1], output_pred[2], output_pred[3]
            y_0, y_1, y_2, y_3 = output_real[0][0], output_real[1][0], output_real[2][0], output_real[3][0]

        if direction_only:
            loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
            loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
            loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
            total_loss = loss_1 + loss_2 + loss_3
        if config['network'] == "vgg":
            abs_loss_0 = torch.mean((y_pred_0 - y_0) ** 2, dim=(1, 2, 3))
            loss_0 = 1 - similarity_loss(y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1))
            abs_loss_1 = torch.mean((y_pred_1 - y_1) ** 2, dim=(1, 2, 3))
            loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
            abs_loss_2 = torch.mean((y_pred_2 - y_2) ** 2, dim=(1, 2, 3))
            loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
            abs_loss_3 = torch.mean((y_pred_3 - y_3) ** 2, dim=(1, 2, 3))
            loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
            total_loss = loss_0 + loss_1 + loss_2 + loss_3 + lamda * (abs_loss_0 + abs_loss_1 + abs_loss_2 + abs_loss_3)
            # only layer4
            # total_loss = loss_3 + lamda * abs_loss_3
            # last two layers
            # total_loss = loss_2 + loss_3 + lamda * (abs_loss_2 + abs_loss_3)
            # only direction loss
            # total_loss = loss_0 + loss_1 + loss_2 + loss_3
            # only MSE loss
            # total_loss = abs_loss_0 + abs_loss_1 + abs_loss_2 + abs_loss_3
        if config['network'] == "resnet18":
            abs_loss_0 = torch.mean((y_pred_0 - y_0) ** 2, dim=(1, 2, 3))
            loss_0 = 1 - similarity_loss(y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1))
            abs_loss_1 = torch.mean((y_pred_1 - y_1) ** 2, dim=(1, 2, 3))
            loss_1 = 1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1))
            abs_loss_2 = torch.mean((y_pred_2 - y_2) ** 2, dim=(1, 2, 3))
            loss_2 = 1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1))
            # abs_loss_3 = torch.mean((y_pred_3 - y_3) ** 2, dim=(1, 2, 3))
            # loss_3 = 1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1))
            total_loss = loss_0 + loss_1 + loss_2 + lamda * (abs_loss_0 + abs_loss_1 + abs_loss_2)

        label_score += list(zip(Y.cpu().data.numpy().tolist(), total_loss.cpu().data.numpy().tolist()))
    
    toc = time.time()-tic
    # print(toc)

    if module == 'test':
        labels, scores = zip(*label_score)
        scores = np.array(scores)
        for i in range(len(scores)):
            if scores[i] < thre:
                print("Normal")
            else:
                print("Abnormal")
        return 100

    elif module == 'val':
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        indx1 = labels == target_class
        indx2 = labels != target_class
        labels[indx1] = 1
        labels[indx2] = 0
        scores = np.array(scores)
        max_score = scores.max()
        min_score = scores.min()
        scores = (scores - min_score) / (max_score - min_score)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 4)

        bad_score_min = 1
        good_score_max = 0
        for i in range(len(scores)):
            if labels[i] == 0:
                if scores[i] < bad_score_min:
                    bad_score_min = scores[i]
            if labels[i] == 1:
                if scores[i] > good_score_max:
                    good_score_max = scores[i]

        thre = (bad_score_min + good_score_max) / 2
        print("thre:", thre)
        print("bad_score_min:", bad_score_min)
        print("good_score_max:", good_score_max)

        return roc_auc

    # thre = 0.38
    # for i in range(len(scores)):
    #     if test_dataloader.dataset.imgs[i][1] == 0 and scores[i] < thre:
    #         print(test_dataloader.dataset.imgs[i][0])
    #         print(scores[i])
    #     if test_dataloader.dataset.imgs[i][1] == 1 and scores[i] > thre:
    #         print(test_dataloader.dataset.imgs[i][0])
    #         print(scores[i])
    # 
    # plt.plot(fpr, tpr, label='SVM model AUC %0.4f' % roc_auc, color='blue', lw=2)
    # plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    



def localization_test(model, vgg, train_dataloader, test_dataloader, ground_truth, config):
    tic2 = time.time()
    localization_method = config['localization_method']
    if localization_method == 'gradients':
        grad = gradients_localization(model, vgg, test_dataloader, config)
    if localization_method == 'smooth_grad':
        grad = smooth_grad_localization(model, vgg, test_dataloader, config)
    if localization_method == 'gbp':
        grad = gbp_localization(model, vgg, test_dataloader, config)
    if localization_method == 'padim':
        grad = padim_localization(model, vgg, train_dataloader, test_dataloader, ground_truth, config)
    threshold, auc = compute_localization_auc(grad, ground_truth)
    toc2 = time.time() - tic2
    print("localization time:", toc2)
    i = 0
    for data in test_dataloader:
        X, Y = data
        if i == 0:
            X_data = X
            Y_data = Y
        else:
            X_data = torch.cat((X_data, X), 0)
            Y_data = torch.cat((Y_data, Y), 0)
        i = i + 1
    test_imgs = np.asarray(X_data)
    gt_mask_list = np.mean(ground_truth, axis=3)
    save_dir = "./{}/{}/{}_image_save/".format(config["experiment_name"], config["mydata_name"], localization_method)
    os.makedirs(save_dir, exist_ok=True)
    # plot_fig(test_imgs, grad, gt_mask_list, threshold, save_dir, config["mydata_name"])
    # gbp_plot(test_imgs, grad, gt_mask_list, config)

    return auc


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def padim_localization(model, model_teacher, train_dataloader, test_dataloader, ground_truth, config):
    model.cuda()
    model.eval()
    model_teacher.cuda()
    model_teacher.eval()

    print("padim Method:")

    # train parameters
    i = 0
    for data in train_dataloader:
        X, Y = data
        if i == 0:
            X_data = X
            Y_data = Y
        else:
            X_data = torch.cat((X_data, X), 0)
            Y_data = torch.cat((Y_data, Y), 0)
        i = i + 1

    x_data = Variable(X_data).cuda()
    if config['network'] == "vgg":
        output_pred = []
        output_real = []
        for data in train_dataloader:
            X, Y = data
            with torch.no_grad():
                output_pred_temp = model.forward(X.cuda())
                output_real_temp = model_teacher(X.cuda())
            output_pred.append(output_pred_temp)
            output_real.append(output_real_temp)

        y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = output_pred[0][1], output_pred[0][3], output_pred[0][6], output_pred[0][9], output_pred[0][12]
        y_1, y_2, y_3, y_4, y_5 = output_real[0][1], output_real[0][3], output_real[0][6], output_real[0][9], output_real[0][12]
        y_pred_vector = y_pred_2
        y_pred_vector = embedding_concat(y_pred_vector, y_pred_3)
        y_pred_vector = embedding_concat(y_pred_vector, y_pred_4.cpu())
        y_vector = y_2
        y_vector = embedding_concat(y_vector, y_3)
        y_vector = embedding_concat(y_vector, y_4.cpu())

    if config['network'] == "resnet18":
        output_pred = model.forward(x_data)
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
            model_teacher(x_data)
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
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[0], output_pred[1], output_pred[2], output_pred[3]
        y_0, y_1, y_2, y_3 = output_real[0][0], output_real[1][0], output_real[2][0], output_real[3][0]
        y_pred_vector = y_pred_0
        y_pred_vector = embedding_concat(y_pred_vector, y_pred_1)
        # y_pred_vector = embedding_concat(y_pred_vector, y_pred_2.cpu())
        # y_pred_vector = embedding_concat(y_pred_vector, y_pred_3.cpu())
        y_vector = y_0
        y_vector = embedding_concat(y_vector, y_1)
        # y_vector = embedding_concat(y_vector, y_2.cpu())

    if config["identity"] == "teacher":
        y_pred_vector = y_vector
    B, C, H, W = y_pred_vector.size()
    embedding_vectors = y_pred_vector.permute(0, 2, 3, 1)
    embedding_vectors = embedding_vectors.reshape(B * H * W, C)
    mean = torch.mean(embedding_vectors, dim=0).detach().numpy()
    I = np.identity(C)
    cov = np.cov(embedding_vectors.detach().numpy(), rowvar=False) + 0.01 * I
    train_out = [mean, cov]

    # model.eval()
    # model_teacher.eval()
    # test parameters
    i = 0
    for data in test_dataloader:
        X, Y = data
        if i == 0:
            X_data = X
            Y_data = Y
        else:
            X_data = torch.cat((X_data, X), 0)
            Y_data = torch.cat((Y_data, Y), 0)
        i = i + 1

    x_data = Variable(X_data).cuda()
    if config['network'] == "vgg":
        output_pred = []
        output_real = []
        for data in test_dataloader:
            X, Y = data
            with torch.no_grad():
                output_pred_temp = model.forward(X.cuda())
                output_real_temp = model_teacher(X.cuda())
            output_pred.append(output_pred_temp)
            output_real.append(output_real_temp)
        y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5 = output_pred[0][1], output_pred[0][3], output_pred[0][6], output_pred[0][9], output_pred[0][12]
        y_1, y_2, y_3, y_4, y_5 = output_real[0][1], output_real[0][3], output_real[0][6], output_real[0][9], output_real[0][12]
        y_pred_vector = y_pred_2
        y_pred_vector = embedding_concat(y_pred_vector, y_pred_3)
        y_pred_vector = embedding_concat(y_pred_vector, y_pred_4.cpu())
        y_vector = y_2
        y_vector = embedding_concat(y_vector, y_3)
        y_vector = embedding_concat(y_vector, y_4.cpu())
    if config['network'] == "resnet18":
        output_pred = model.forward(x_data)
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
            model_teacher(x_data)
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
        y_pred_0, y_pred_1, y_pred_2, y_pred_3 = output_pred[0], output_pred[1], output_pred[2], output_pred[3]
        y_0, y_1, y_2, y_3 = output_real[0][0], output_real[1][0], output_real[2][0], output_real[3][0]
        y_pred_vector = y_pred_0
        y_pred_vector = embedding_concat(y_pred_vector, y_pred_1)
        # y_pred_vector = embedding_concat(y_pred_vector, y_pred_2.cpu())
        # y_pred_vector = embedding_concat(y_pred_vector, y_pred_3.cpu())
        y_vector = y_0
        y_vector = embedding_concat(y_vector, y_1)

    if config["identity"] == "teacher":
        y_pred_vector = y_vector

    B, C, H, W = y_pred_vector.size()
    embedding_vectors = y_pred_vector.permute(0, 2, 3, 1)
    embedding_vectors = embedding_vectors.reshape(B * H * W, C).detach().numpy()
    dist_list = []
    mean_val = train_out[0]
    conv_inv_val = np.linalg.inv(train_out[1])
    embedding_vectors = embedding_vectors[:, ] - mean_val
    embedding_vectors = torch.tensor(embedding_vectors)
    conv_inv_val = torch.tensor(conv_inv_val, dtype=torch.float32)
    dist = [torch.mm(torch.mm(sample.reshape(1, C), conv_inv_val), sample.reshape(1, C).t()) for sample in embedding_vectors]
    dist_list.append(dist)
    dist_list = np.array(dist_list).reshape(B, H, W)

    dist_list = torch.tensor(dist_list.astype(float))
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x_data.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(Y_data)
    target_class = config["normal_class"]
    indx1 = gt_list == target_class
    indx2 = gt_list != target_class
    gt_list[indx1] = 0
    gt_list[indx2] = 1
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('padim method: image ROCAUC: %.5f' % (img_roc_auc))

    return scores


def grad_calc(inputs, model, vgg, config):
    inputs = inputs.cuda()
    inputs.requires_grad = True
    temp = torch.zeros(inputs.shape)
    lamda = config['lamda']
    criterion = nn.MSELoss()
    similarity_loss = torch.nn.CosineSimilarity()

    for i in range(inputs.shape[0]):
        output_pred = model.forward(inputs[i].unsqueeze(0), target_layer=14)
        output_real = vgg(inputs[i].unsqueeze(0))
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]
        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
        model.zero_grad()
        total_loss.backward()

        temp[i] = inputs.grad[i]

    return temp


def gradients_localization(model, vgg, test_dataloader, config):
    model.eval()
    print("Vanilla Backpropagation:")
    temp = None
    for data in test_dataloader:
        X, Y = data
        grad = grad_calc(X, model, vgg, config)
        temp = np.zeros((grad.shape[0], grad.shape[2], grad.shape[3]))
        for i in range(grad.shape[0]):
            grad_temp = convert_to_grayscale(grad[i].cpu().numpy())
            grad_temp = grad_temp.squeeze(0)
            grad_temp = gaussian_filter(grad_temp, sigma=4)
            temp[i] = grad_temp
    return temp


class VanillaSaliency():
    def __init__(self, model, vgg, device, config):
        self.model = model
        self.vgg = vgg
        self.device = device
        self.config = config
        self.model.eval()

    def generate_saliency(self, data, make_single_channel=True):
        data_var_sal = Variable(data).to(self.device)
        self.model.zero_grad()
        if data_var_sal.grad is not None:
            data_var_sal.grad.data.zero_()
        data_var_sal.requires_grad_(True)

        lamda = self.config['lamda']
        criterion = nn.MSELoss()
        similarity_loss = torch.nn.CosineSimilarity()

        output_pred = self.model.forward(data_var_sal)
        output_real = self.vgg(data_var_sal)
        y_pred_1, y_pred_2, y_pred_3 = output_pred[6], output_pred[9], output_pred[12]
        y_1, y_2, y_3 = output_real[6], output_real[9], output_real[12]

        abs_loss_1 = criterion(y_pred_1, y_1)
        loss_1 = torch.mean(1 - similarity_loss(y_pred_1.view(y_pred_1.shape[0], -1), y_1.view(y_1.shape[0], -1)))
        abs_loss_2 = criterion(y_pred_2, y_2)
        loss_2 = torch.mean(1 - similarity_loss(y_pred_2.view(y_pred_2.shape[0], -1), y_2.view(y_2.shape[0], -1)))
        abs_loss_3 = criterion(y_pred_3, y_3)
        loss_3 = torch.mean(1 - similarity_loss(y_pred_3.view(y_pred_3.shape[0], -1), y_3.view(y_3.shape[0], -1)))
        total_loss = loss_1 + loss_2 + loss_3 + lamda * (abs_loss_1 + abs_loss_2 + abs_loss_3)
        self.model.zero_grad()
        total_loss.backward()
        grad = data_var_sal.grad.data.detach().cpu()

        if make_single_channel:
            grad = np.asarray(grad.detach().cpu().squeeze(0))
            # grad = max_regarding_to_abs(np.max(grad, axis=0), np.min(grad, axis=0))
            # grad = np.expand_dims(grad, axis=0)
            grad = convert_to_grayscale(grad)
            # print(grad.shape)
        else:
            grad = np.asarray(grad)
        return grad


def generate_smooth_grad(data, param_n, param_sigma_multiplier, vbp, single_channel=True):
    smooth_grad = None

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(data) - torch.min(data)).item()
    VBP = vbp
    for x in range(param_n):
        noise = Variable(data.data.new(data.size()).normal_(mean, sigma ** 2))
        noisy_img = data + noise
        vanilla_grads = VBP.generate_saliency(noisy_img, single_channel)
        if not isinstance(vanilla_grads, np.ndarray):
            vanilla_grads = vanilla_grads.detach().cpu().numpy()
        if smooth_grad is None:
            smooth_grad = vanilla_grads
        else:
            smooth_grad = smooth_grad + vanilla_grads

    smooth_grad = smooth_grad / param_n
    return smooth_grad


class IntegratedGradients():
    def __init__(self, model, vgg, device):
        self.model = model
        self.vgg = vgg
        self.gradients = None
        self.device = device
        # Put model in evaluation mode
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        step_list = np.arange(steps + 1) / steps
        xbar_list = [input_image * step for step in step_list]
        return xbar_list

    def generate_gradients(self, input_image, make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model, self.vgg, self.device)
        saliency = vanillaSaliency.generate_saliency(input_image, make_single_channel)
        if not isinstance(saliency, np.ndarray):
            saliency = saliency.detach().cpu().numpy()
        return saliency

    def generate_integrated_gradients(self, input_image, steps, make_single_channel=True):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = None
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, False)
            if integrated_grads is None:
                integrated_grads = deepcopy(single_integrated_grad)
            else:
                integrated_grads = (integrated_grads + single_integrated_grad)
        integrated_grads /= steps
        saliency = integrated_grads[0]
        img = input_image.detach().cpu().numpy().squeeze(0)
        saliency = np.asarray(saliency) * img
        if make_single_channel:
            saliency = max_regarding_to_abs(np.max(saliency, axis=0), np.min(saliency, axis=0))
        return saliency


def generate_integrad_saliency_maps(model, vgg, preprocessed_image, device, steps=100, make_single_channel=True):
    IG = IntegratedGradients(model, vgg, device)
    integrated_grads = IG.generate_integrated_gradients(preprocessed_image, steps, make_single_channel)
    if make_single_channel:
        integrated_grads = convert_to_grayscale(integrated_grads)
    return integrated_grads


class GuidedBackprop():
    def __init__(self, model, vgg, device):
        self.model = model
        self.vgg = vgg
        self.gradients = None
        self.forward_relu_outputs = []
        self.device = device
        self.hooks = []
        self.model.eval()
        self.update_relus()

    def update_relus(self):

        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.model.modules():
            if isinstance(module, ReLU):
                self.hooks.append(module.register_backward_hook(relu_backward_hook_function))
                self.hooks.append(module.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self, input_image, config, make_single_channel=True):
        vanillaSaliency = VanillaSaliency(self.model, self.vgg, self.device, config=config)
        sal = vanillaSaliency.generate_saliency(input_image, make_single_channel)
        if not isinstance(sal, np.ndarray):
            sal = sal.detach().cpu().numpy()
        for hook in self.hooks:
            hook.remove()
        return sal


def gbp_localization(model, vgg, test_dataloader, config):
    model.eval()
    print("GBP Method:")

    grad1 = None
    i = 0

    # ori_img = []

    for data in test_dataloader:
        X, Y = data
        grad1 = np.zeros((X.shape[0], 1, 128, 128), dtype=np.float32)
        for x in X:
            x_data = x.view(1, 3, 128, 128)
            # ori_img.append(data)

            GBP = GuidedBackprop(model, vgg, 'cuda:0')
            gbp_saliency = abs(GBP.generate_gradients(x_data, config))
            gbp_saliency = (gbp_saliency - min(gbp_saliency.flatten())) / (
                    max(gbp_saliency.flatten()) - min(gbp_saliency.flatten()))
            saliency = gbp_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[i] = saliency
            i += 1

    grad1 = grad1.reshape(-1, 128, 128)

    # for i in range(6, 10):
    #     x = ori_img[i].view(-1, 128, 128)
    #     x = x.permute(1, 2, 0).numpy()
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(x)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(grad1[i])
    #     # plt.savefig(str(i)+'.png', bbox_inches='tight')
    #     # plt.show()
    #     plt.close()

    return grad1


def smooth_grad_localization(model, vgg, test_dataloader, config):
    model.eval()
    print("Smooth Grad Method:")

    grad1 = None
    i = 0

    for data in test_dataloader:
        X, Y = data
        grad1 = np.zeros((X.shape[0], 1, 128, 128), dtype=np.float32)
        for x in X:
            data = x.view(1, 3, 128, 128)

            vbp = VanillaSaliency(model, vgg, 'cuda:0', config)

            smooth_grad_saliency = abs(generate_smooth_grad(data, 50, 0.05, vbp))
            smooth_grad_saliency = (smooth_grad_saliency - min(smooth_grad_saliency.flatten())) / (
                    max(smooth_grad_saliency.flatten()) - min(smooth_grad_saliency.flatten()))
            saliency = smooth_grad_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[i] = saliency
            i += 1

    grad1 = grad1.reshape(-1, 128, 128)
    return grad1


def compute_localization_auc(grad, x_ground):
    x_ground_comp = np.mean(x_ground, axis=3)
    ggg = grad.flatten()
    xxx = x_ground_comp.flatten()
    xxx[xxx > 0.5] = 1
    xxx[xxx < 0.5] = 0
    precision, recall, thresholds = precision_recall_curve(xxx, ggg)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    fpr, tpr, _ = roc_curve(xxx, ggg)
    
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, label='my model AUC %0.5f' % roc_auc, color='blue', lw=2)
    # plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating Curve')
    # plt.legend(loc="lower right")
    # plt.close()

    return threshold, auc(fpr, tpr)


def gbp_plot(test_img, grad, gts, config):
    for i in range(len(gts)):
        image_save_path = "local_equal_net/{}/gbp_image_save".format(config["mydata_name"])
        os.makedirs(image_save_path, exist_ok=True)
        image_save_path = image_save_path + '/'
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(test_img[i].transpose(1, 2, 0))
        plt.subplot(1, 3, 2)
        plt.imshow(gts[i])
        plt.subplot(1, 3, 3)
        plt.imshow(grad[i])
        plt.savefig(image_save_path + "%03d" % i + '.png', bbox_inches='tight')
        plt.close()
        # plt.show()


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x