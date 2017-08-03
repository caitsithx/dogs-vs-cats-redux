import glob
import os

import bcolz
import torch
import torch.nn as nn
from torchvision import models

from settings import output_num, MODEL_DIR

w_files_training = []


def get_acc_from_w_filename(filename):
    try:
        stracc = filename.split('_')[-2]
        return float(stracc)
    except:
        return 0.


def load_best_weights(model):
    w_files = glob.glob(os.path.join(MODEL_DIR, model.name) + '_*.pth')
    max_acc = 0
    best_file = None
    for w_file in w_files:
        try:
            stracc = w_file.split('_')[-2]
            acc = float(stracc)
            if acc > max_acc:
                best_file = w_file
                max_acc = acc
            w_files_training.append((acc, w_file))
        except:
            continue
    if max_acc > 0:
        print('loading weight: {}'.format(best_file))
        model.load_state_dict(torch.load(best_file))


def save_weights(acc, model, epoch, max_num=2):
    f_name = '{}_{}_{:.5f}_.pth'.format(model.name, epoch, acc)
    w_file_path = os.path.join(MODEL_DIR, f_name)
    if len(w_files_training) < max_num:
        w_files_training.append((acc, w_file_path))
        torch.save(model.state_dict(), w_file_path)
        return
    min = 10.0
    index_min = -1
    for i, item in enumerate(w_files_training):
        val_acc, fp = item
        if min > val_acc:
            index_min = i
            min = val_acc
    # print(min)
    if acc > min:
        torch.save(model.state_dict(), w_file_path)
        try:
            os.remove(w_files_training[index_min][1])
        except:
            print('Failed to delete file: {}'.format(w_files_training[index_min][1]))
        w_files_training[index_min] = (acc, w_file_path)


def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def load_weights_file(model, w_file):
    model.load_state_dict(torch.load(w_file))


def create_model1(arch, pretrained=True):
    if pretrained:
        print("=> using pre-trained model '{}'".format(arch))
    else:
        print("=> creating model '{}'".format(arch))

    model = models.__dict__[arch](pretrained=pretrained)

    if arch.startswith('resnet') or arch.startswith("inception"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_num)
    elif arch.startswith("desnet"):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_num)
    elif arch.startswith('vgg'):
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_num))

    if arch.startswith("inception_v3"):
        model.aux_logits = False

    model = model.cuda()

    if arch.startswith('resnet') or arch.startswith("inception"):
        model.batch_size = 56
    elif arch.startswith('vgg') or arch.startswith("desnet"):
        model.batch_size = 24

    model.name = arch

    return model

    # if arch.startswith('alexnet') or arch.startswith('vgg'):
    #     model.features = torch.nn.DataParallel(model.features)
    #     model.cuda()
    # else:
    #     model = torch.nn.DataParallel(model).cuda()


# def create_res50(load_weights=False):
# def create_res101(load_weights=False):
# def create_res152(load_weights=False):
# def create_dense161(load_weights=False):
# def create_dense169(load_weights=False):
# def create_dense121(load_weights=False):
# def create_dense201(load_weights=False):
# def create_vgg19bn(load_weights=False):
# def create_vgg16bn(load_weights=False):
# def create_inceptionv3(load_weights=False):
# def create_inceptionresv2(load_weights=False):

def create_model(model_name):
    return create_model1(model_name)
