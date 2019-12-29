#!coding:utf-8
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms

from utils.data_utils import NO_LABEL

load = {}
def register_dataset(dataset):
    def warpper(f):
        load[dataset] = f
        return f
    return warpper

def encode_label(label):
    return NO_LABEL* (label +1)

def decode_label(label):
    return NO_LABEL * label -1

def split_relabel_data(np_labs, labels, label_per_class,
                        num_classes):
    """ Return the labeled indexes and unlabeled_indexes
    """
    labeled_idxs = []
    unlabed_idxs = []
    for id in range(num_classes):
        indexes = np.where(np_labs==id)[0]
        np.random.shuffle(indexes)
        labeled_idxs.extend(indexes[:label_per_class])
        unlabed_idxs.extend(indexes[label_per_class:])
    np.random.shuffle(labeled_idxs)
    np.random.shuffle(unlabed_idxs)
    ## relabel dataset
    for idx in unlabed_idxs:
        labels[idx] = encode_label(labels[idx])

    return labeled_idxs, unlabed_idxs
     

@register_dataset('cifar10')
def cifar10(n_labels, data_root='./data-local/cifar10/'):
    channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                         std = [0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR10(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR10(data_root, train=False, download=True,
                                   transform=eval_transform)
    num_classes = 10
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.train_labels),
                                    trainset.train_labels,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'label_idxs': labeled_idxs,
        'unlab_idxs': unlabed_idxs,
        'num_classes': num_classes
    }


@register_dataset('cifar100')
def cifar100(n_labels, data_root='./data-local/cifar100/'):
    channel_stats = dict(mean = [0.5071, 0.4867, 0.4408],
                         std = [0.2675, 0.2565, 0.2761])
    train_transform = transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    trainset = tv.datasets.CIFAR100(data_root, train=True, download=True,
                                   transform=train_transform)
    evalset = tv.datasets.CIFAR100(data_root, train=False, download=True,
                                   transform=eval_transform)
    num_classes = 100
    label_per_class = n_labels // num_classes
    labeled_idxs, unlabed_idxs = split_relabel_data(
                                    np.array(trainset.train_labels),
                                    trainset.train_labels,
                                    label_per_class,
                                    num_classes)
    return {
        'trainset': trainset,
        'evalset': evalset,
        'labeled_idxs': labeled_idxs,
        'unlabeled_idxs': unlabed_idxs,
        'num_classes': num_classes
    }
