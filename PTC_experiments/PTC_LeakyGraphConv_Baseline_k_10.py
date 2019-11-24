import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import torch.nn.functional as F

from data_reader.cross_validation_reader import getcross_validation_split
from model.LeakyGraphConvNet import GraphConv_GNN
from impl.binGraphClassifier_Graph_Conv import modelImplementation_GraphBinClassifier
from utils.utils import printParOnFile

if __name__ == '__main__':

    n_epochs = 100
    n_classes = 2
    n_units = 50
    lr = 0.001
    drop_prob=0.50
    weight_decay=5e-4
    momentum = 0.9
    batch_size = 16
    n_folds = 10
    test_epoch = 1
    max_k = 10
    aggregator = 'concat'
    test_name = "Leaky_DeepGraphConv_linear_Baseline_Test"

    dataset_path = '~/Dataset/PTC_MR'
    dataset_name = 'PTC_MR'

    test_name = test_name + "_data-" + dataset_name + "_aggregator-" + aggregator + "_nFold-" + str(
        n_folds) + "_lr-" + str(lr)+"_drop_prob-"+str(drop_prob)+"_weight-decay-"+ str(weight_decay) + "_batchSize-" + \
                str(batch_size) + "_nHidden-" + str(n_units) + "_maxK-" + str(max_k)
    training_log_dir = os.path.join("./test_log/", test_name)
    if not os.path.exists(training_log_dir):
        os.makedirs(training_log_dir)

    printParOnFile(test_name=test_name, log_dir=training_log_dir, par_list={"dataset_name": dataset_name,
                                                                            "n_fold": n_folds,
                                                                            "learning_rate": lr,
                                                                            "drop_prob": drop_prob,
                                                                            "weight_decay": weight_decay,
                                                                            "batch_size": batch_size,
                                                                            "test_epoch": test_epoch,
                                                                            "n_hidden": n_units,
                                                                            "max_k": max_k,
                                                                            "aggregator": aggregator})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.NLLLoss()

    dataset_cv_splits = getcross_validation_split(dataset_path, dataset_name, n_folds, batch_size)
    for split_id, split in enumerate(dataset_cv_splits):
        loader_train = split[0]
        loader_test = split[1]
        loader_valid = split[2]

        model = GraphConv_GNN(loader_train.dataset.num_node_labels, n_units, n_classes,drop_prob=drop_prob, max_k=max_k).to(device)

        model_impl = modelImplementation_GraphBinClassifier(model, lr, criterion, device).to(device)

        model_impl.set_optimizer(weight_decay=weight_decay)

        model_impl.train_test_model(split_id, loader_train, loader_test, loader_valid, n_epochs, test_epoch, aggregator,
                                    test_name, training_log_dir)
