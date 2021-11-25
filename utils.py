import re, sys, os
from os.path import basename
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn import metrics

def path2age(path, pat, pos):
    return int(re.split(pat, basename(path))[pos])

def accuracy(preds, labels):
    return (preds.squeeze()==labels.squeeze()).float().mean()

def accuracy_percent(preds, labels):
    return (preds.squeeze()==labels.squeeze()).float().mean().mul_(100.0)


def multi_accuracies(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def erase_print(content):
    sys.stdout.write('\033[2K\033[1G')
    sys.stdout.write(content)
    sys.stdout.flush()

def mkdir_p(path):
    try:
        os.makedirs(os.path.abspath(path))
    except OSError as exc: 
        if exc.errno == os.errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def get_ctx(n):
    return torch.device(f'cuda:{n}') if n >=0 else torch.device('cpu')

class Recorder():
    def __init__(self):
        self.N = 0
        self.loss = 0
        self.n_crct = 0

    def reset(self):
        self.N = 0
        self.loss = 0
        self.n_crct = 0

    def gulp(self, n, loss, acc):
        self.N += n
        self.loss += n*loss
        self.n_crct += int(n*acc)

    def excrete(self):
        self.loss = self.loss / self.N
        self.acc = self.n_crct / self.N
        return self
    
    def result_as_string(self):
        return f'{self.N}, {self.loss:.4f}, {self.acc:.4f}'


def calc_coeff(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def l2_rec(src, trg):
    return torch.sum((src - trg)**2) / (src.shape[0] * src.shape[1])


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy.mean() 



def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels


def get_features(net, train_loader, embedding=True, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        train_loader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(train_loader, desc="extracting all features from dataset")
    else:
        train_bar = train_loader
    for step, (batch_imgs, batch_lbls, _) in enumerate(train_bar):
        batch_features = net(batch_imgs.cuda(), emb=embedding)
        features.append(batch_features.cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)


def calculate_roc_auc_eer(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    similarity_scores = np.sum(embeddings1 * embeddings2, axis=1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                                                                threshold,
                                                                similarity_scores[train_set],
                                                                actual_issame[train_set]
                                                                )
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        # calculate tpr and fpr under varing thresholds on current test subset
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                                                                                                threshold,
                                                                                                similarity_scores[test_set],
                                                                                                actual_issame[test_set]
                                                                                                )
        # calculate the face verificatrion accuracy on current test subset using best threshold from train subset                                                                                                 
        _, _, accuracy[fold_idx] = calculate_accuracy(
                                                    thresholds[best_threshold_index], 
                                                    similarity_scores[test_set], 
                                                    actual_issame[test_set]
                                                    )
    # calculate mean performance on K folds
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
  #  print(min(tpr))
  #  print(max(tpr))
  #  print(min(fpr))
  #  print(max(fpr))
  #  os._exit(0)

    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), min(fpr), max(tpr))
   # eer = 0.01

    mean_acc = accuracy.mean()
    mean_best_threshold = best_thresholds.mean()
    return tpr, fpr, auc, eer, mean_acc, mean_best_threshold



def calculate_accuracy(threshold, sim_scores, actual_issame):
    predict_issame = np.greater_equal(sim_scores, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / sim_scores.size
    return tpr, fpr, acc


def calculate_accuracy_euclidian_distance(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def roc_curve_plot(fpr, tpr, save_path):
    """Create a roc_curve plot and save."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)



def calculate_roc_auc_eer_euclidian_distance(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy_euclidian_distance(
                                                                threshold,
                                                                dist[train_set],
                                                                actual_issame[train_set]
                                                                )
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        # calculate tpr and fpr under varing thresholds on current test subset
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy_euclidian_distance(
                                                                                                threshold,
                                                                                                dist[test_set],
                                                                                                actual_issame[test_set]
                                                                                                )
        # calculate the face verificatrion accuracy on current test subset using best threshold from train subset                                                                                                 
        _, _, accuracy[fold_idx] = calculate_accuracy_euclidian_distance(
                                                    thresholds[best_threshold_index], 
                                                    dist[test_set], 
                                                    actual_issame[test_set]
                                                    )
    # calculate mean performance on K folds
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
  #  print(min(tpr))
  #  print(max(tpr))
  #  print(min(fpr))
  #  print(max(fpr))
  #  os._exit(0)

    auc = metrics.auc(fpr, tpr)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), min(fpr), max(tpr))
   
    mean_acc = accuracy.mean()
    mean_best_threshold = best_thresholds.mean()
    return tpr, fpr, auc, eer, mean_acc, mean_best_threshold


def argmax_mae(output, target):
    """age range, categorized by [0-12, 13-18, 19-25,
    26-35, 36-45, 46-55, 56-65, >= 66]"""
    with torch.no_grad():
        batch_size = output.size(0)
        predicted_age_ranges=torch.argmax(output,1)
        mae=torch.sum(torch.abs(predicted_age_ranges-target)).float()/batch_size
        
        return mae