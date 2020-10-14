import numpy as np
import torch
import torch.nn as nn
import time
from collections import OrderedDict
from BuildModelResidual import CreateNetworkResidual
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

from thop import profile

from tabulate import tabulate
import os
import pickle

from copy import deepcopy

def add_random_block_residual(list_blocks, tried_blocks, output_size, downsampling=False):
    min_feat_maps = 8
    max_feat_maps = 64
    min_residual_layers = 2
    max_residual_layers = 6

    if len(list_blocks) > 0:
        prev_feat_maps = list_blocks[-1]["Num. Feature Maps"]
    else:
        prev_feat_maps = 0

    is_block_used_before = True

    while is_block_used_before:
        feat_maps = np.random.randint(min_feat_maps, max_feat_maps)

        while feat_maps == prev_feat_maps:
            # This is done to avoid two consecutive blocks with the same number of feature maps
            # which will result in a missing shortcut layer in the second block.
            feat_maps = np.random.randint(min_feat_maps, max_feat_maps)

        residual_layers = np.random.randint(min_residual_layers, max_residual_layers)
        r = np.random.rand()

        if downsampling:
            if r < 0.6 and output_size >= 4:
                downsample = True
                output_size = int(output_size / 2)
            else:
                downsample = False
        else:
            downsample = False

        residual_block = {"Num. Feature Maps": feat_maps, "Num. Residual Layers": residual_layers,
                          "Downsampling": downsample}

        if residual_block not in tried_blocks:
            is_block_used_before = False
        else:
            print("\t\tBlock tested before...")

    tried_blocks.append(residual_block)
    list_blocks.append(residual_block)

    return list_blocks, tried_blocks, output_size


class DNNDeepening:
    def __init__(self, deepening_par):
        self.list_blocks = []
        self.model = None
        self.flops = None
        self.dataset_name = deepening_par.dataset_name
        self.input_size = deepening_par.input_size
        self.output_size = deepening_par.input_size[-1]
        self.temp_weights_path = deepening_par.temp_weights_path
        self.G_trial = deepening_par.G_trial
        self.deepening_results = None

        if self.dataset_name == "CIFAR10":
            self.num_classes = 10
        elif self.dataset_name == "CIFAR100":
            self.num_classes = 100
        else:
            self.num_classes = 1

    def fit(self, train_loader, validation_loader):
        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            self.fit_medical_data(train_loader, validation_loader)
        else:
            self.fit_non_medical_data(train_loader, validation_loader)

    def fit_medical_data(self, train_loader, validation_loader):
        new_val_auc = 0
        best_val_auc = 0
        best_dnn_valid_fpr = []
        best_dnn_valid_tpr = []

        fig, ax = plt.subplots(1, 1)

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')

        i = 0
        no_change = 0
        while no_change < 2:
            i += 1
            print("Iteration: " + str(i))
            dnn_test = deepcopy(self.list_blocks)

            already_tried_blocks = []

            for j in range(self.G_trial):
                print("\tTrial: " + str(j))
                dnn_temp = deepcopy(dnn_test)

                if i == 1:
                    dnn_temp, already_tried_blocks, self.output_size = add_random_block_residual(dnn_temp,
                                                                                                 already_tried_blocks,
                                                                                                 self.output_size,
                                                                                                 downsampling=False)

                    model, test_acc, test_loss, test_auc, dnn_valid_fpr, dnn_valid_tpr,\
                    dnn_valid_sensitivity, dnn_valid_specificity =\
                        self.evaluate_medical(dnn_temp, self.input_size, train_loader, validation_loader,
                                              i, load_weights=False)
                else:
                    dnn_temp, already_tried_blocks, self.output_size = add_random_block_residual(dnn_temp,
                                                                                                 already_tried_blocks,
                                                                                                 self.output_size,
                                                                                                 downsampling=True)

                    model, test_acc, test_loss, test_auc, dnn_valid_fpr, dnn_valid_tpr,\
                    dnn_valid_sensitivity, dnn_valid_specificity =\
                        self.evaluate_medical(dnn_temp, self.input_size, train_loader, validation_loader,
                                              i, load_weights=True)

                if j == 0:
                    self.list_blocks = deepcopy(dnn_temp)
                    best_auc = test_auc
                    best_dnn_valid_fpr = dnn_valid_fpr
                    best_dnn_valid_tpr = dnn_valid_tpr
                    new_val_auc = test_auc
                    best_model = deepcopy(model)

                    print("\t\tCurrent best architecture: ")
                    print("\t\t" + str(self.list_blocks))
                    print("\t\tCurrent Validation AUC: " + str(best_auc))
                elif j > 0 and test_auc > best_auc:
                    self.list_blocks = deepcopy(dnn_temp)
                    best_auc = test_auc
                    best_dnn_valid_fpr = dnn_valid_fpr
                    best_dnn_valid_tpr = dnn_valid_tpr
                    new_val_auc = test_auc
                    best_model = deepcopy(model)

                    print("\t\tNew best architecture found: ")
                    print("\t\t" + str(self.list_blocks))
                    print("\t\tCurrent Validation AUC: " + str(best_auc))

            if new_val_auc > best_val_auc:
                best_val_auc = new_val_auc
                no_change = 0
            else:
                no_change += 1

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.plot(best_dnn_valid_fpr, best_dnn_valid_tpr, lw=1,
                    label='Iteration %i - AUC = %0.4f' % (i, best_auc))

            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            figure_path = "./dnn_deepening_results/" + self.dataset_name + "/graphs/pdf/"

            if not os.path.exists(figure_path):
                os.makedirs(figure_path)
            
            plt.savefig(figure_path + "ROC-AUC-Curves-" + str(i) + ".pdf", format='pdf',
                        bbox_inches='tight')

            #ax.set_xlim([0.0, 0.5])
            #ax.set_ylim([0.6, 1.05])
            #plt.savefig(figura_path + "ROC-AUC-Zoomed-" + str(i) + ".pdf", format='pdf',
            #            bbox_inches='tight')

            print("")
            print("\tBest validation AUC: " + str(best_val_auc))
            print("\tNew validation AUC: " + str(new_val_auc))
            torch.save(best_model.state_dict(), self.temp_weights_path)
            print("Weights saved...")

            self.flops, _ = profile(best_model.float(), input_size=self.input_size)

    def evaluate_medical(self, dnn_representation, input_size, train_loader, validation_loader, iteration,
                 load_weights):
        self.model = CreateNetworkResidual(dnn_representation, input_size)

        print(dnn_representation)

        self.model.cuda()
        self.model.half()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        criterion = nn.BCEWithLogitsLoss().cuda()
        criterion.half()

        if load_weights:
            state_dict = torch.load(self.temp_weights_path)
            new_state_dict = OrderedDict()

            for key, value in state_dict.items():
                if key != 'linear.weight' and key != 'linear.bias':
                    new_state_dict[key] = value

            self.model.load_state_dict(new_state_dict, strict=False)

        epochs_no_improve = 0
        epoch = 0

        best_val_loss = 0
        best_val_auc = 0
        best_val_acc = 0
        best_val_fpr = 0
        best_val_tpr = 0
        best_val_sensitivity = 0
        best_val_specificity = 0
        best_model = None

        while epochs_no_improve < 5:
            epoch += 1
            print("\tEpoch [{}]:".format(epoch))

            # phase --> 'training'
            train_loss, _, _, _, _, _, _ = self.one_iter_train_medical(train_loader, iteration,
                                                                       optimizer,
                                                                       criterion,
                                                                       phase='training')

            print("\t\tTraining Loss: {:.4f}".format(train_loss))

            if epoch % 1 == 0:
                # phase --> 'validation'
                val_loss, val_acc, val_auc, val_fpr, val_tpr, val_sensitivity, val_specificity = \
                    self.one_iter_train_medical(validation_loader, iteration, optimizer, criterion, phase='test')

                print("\t\tValidation Loss: {:.4f}, Validation Acc: {:.4f}".format(val_loss, val_acc))
                print("\t\tValidation AUC: {:.4f}".format(val_auc))
                print("\t\tValidation Sensitivity: {:.4f}".format(val_sensitivity))
                print("\t\tValidation Specificity: {:.4f}".format(val_specificity))

                if val_auc > best_val_auc:
                    epochs_no_improve = 0
                    best_val_auc = val_auc
                    best_val_acc = val_acc.item()
                    best_val_loss = val_loss.item()
                    best_val_fpr = val_fpr
                    best_val_tpr = val_tpr
                    best_val_sensitivity = val_sensitivity
                    best_val_specificity = val_specificity
                    best_model = deepcopy(self.model)
                else:
                    epochs_no_improve += 1

        return best_model, best_val_acc, best_val_loss, best_val_auc, best_val_fpr, best_val_tpr,\
               best_val_sensitivity, best_val_specificity

    def one_iter_train_medical(self, data_loader, iteration, optimizer, criterion, phase='training'):
        running_loss = 0.0
        data_size = 0
        y_true = []
        y_pred = []

        for _, (images, labels) in enumerate(data_loader):
            images = images.cuda().half()
            labels = labels.cuda().half()
            data_size += len(labels)

            labels = labels.unsqueeze(1)

            if phase == 'training':
                self.model.train(True)
                # Forward pass
                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()
            else:
                y_true.extend(labels.squeeze().cpu().numpy())
                self.model.eval()
                with torch.no_grad():
                    images = Variable(images)
                    labels = Variable(labels)

                    optimizer.zero_grad()

                    outputs = self.model(images)

                    out_probs = torch.sigmoid(outputs)
                    y_pred.extend(out_probs.squeeze().cpu().numpy())

                    loss = criterion(outputs, labels)

            running_loss += loss * labels.size(0)

        loss = running_loss / data_size

        if phase != 'training':
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            best_threshold_idx = np.argmax(tpr*(1-fpr))
            sensitivity = tpr[best_threshold_idx]
            specificity = 1 - fpr[best_threshold_idx]
            decision_threshold = np.array([0.5]) # Threshold of 0.5 is used only to compute the accuracy

            preds = np.where(y_pred >= decision_threshold, 1.0, 0.0)
            corrects = np.sum(preds == y_true)
            acc = corrects / len(preds)

            AUC = auc(fpr, tpr)
        else:
            fpr, tpr = [], []
            sensitivity = None
            specificity = None
            AUC = 0
            acc = None

        return loss, acc, AUC, fpr, tpr, sensitivity, specificity

    ##################################################################################################################

    def fit_non_medical_data(self, train_loader, validation_loader):
        fig, ax = plt.subplots(1, 1)

        ax.set_xlabel('Deepening Iteration')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Deepening Results on ' + self.dataset_name)

        best_val_acc_so_far = 0
        best_valid_acc_all_iter = []
        i = 0
        no_change = 0
        while no_change < 2:
            i += 1
            print("Iteration: " + str(i))
            dnn_test = deepcopy(self.list_blocks)

            already_tried_blocks = []

            for j in range(self.G_trial):
                print("\tTrial: " + str(j))
                dnn_temp = deepcopy(dnn_test)

                if i == 1:
                    dnn_temp, already_tried_blocks, self.output_size = add_random_block_residual(dnn_temp,
                                                                                                 already_tried_blocks,
                                                                                                 self.output_size,
                                                                                                 downsampling=False)

                    model, test_acc, test_loss = self.evaluate_non_medical(dnn_temp, self.input_size, train_loader,
                                                                           validation_loader, i,
                                                                           load_weights=False)
                else:
                    dnn_temp, already_tried_blocks, self.output_size = add_random_block_residual(dnn_temp,
                                                                                                 already_tried_blocks,
                                                                                                 self.output_size,
                                                                                                 downsampling=True)

                    model, test_acc, test_loss = self.evaluate_non_medical(dnn_temp, self.input_size, train_loader,
                                                                           validation_loader, i,
                                                                           load_weights=True)

                if j == 0:
                    self.list_blocks = deepcopy(dnn_temp)
                    best_acc = test_acc
                    best_model = deepcopy(model)

                    print("\t\tCurrent best architecture: ")
                    print("\t\t" + str(self.list_blocks))
                    print("\t\tCurrent best validation accuracy on this iteration: " + str(best_acc))

                elif j > 0 and test_acc > best_acc:
                    self.list_blocks = deepcopy(dnn_temp)
                    best_acc = test_acc
                    best_model = deepcopy(model)

                    print("\t\tNew best architecture found: ")
                    print("\t\t" + str(self.list_blocks))
                    print("\t\tCurrent best validation accuracy on this iteration: " + str(best_acc))

            best_valid_acc_all_iter.append(best_acc)

            print("")
            print("\tBest validation accuracy found on this iteration: " + str(best_acc))

            if best_acc > best_val_acc_so_far:
                best_val_acc_so_far = best_acc
                no_change = 0
            else:
                no_change += 1

            print("\tBest validation accuracy so far: " + str(best_val_acc_so_far))

            if not os.path.exists(self.temp_weights_path):
                os.makedirs(self.temp_weights_path)

            torch.save(best_model.state_dict(), self.temp_weights_path + "model_weights_during_deepening.pth")
            print("Weights saved...")

            self.flops, _ = profile(best_model.float(), input_size=self.input_size)

            ax.set_xticks(range(i))
            ax.plot(best_valid_acc_all_iter, color="blue", marker='o', linestyle='dashed')

            figure_path = "./dnn_deepening_results/" + self.dataset_name + "/graphs/pdf/"

            if not os.path.exists(figure_path):
                os.makedirs(figure_path)

            plt.savefig(figure_path + "acc_evolution.pdf",
                        format='pdf',
                        bbox_inches='tight')

            with open(figure_path + "valid_acc_all_iter.pickle", "wb") as f:
                pickle.dump(best_valid_acc_all_iter, f)

    def evaluate_non_medical(self, dnn_representation, input_size, train_loader, validation_loader,
                             iteration, load_weights):

        self.model = CreateNetworkResidual(dnn_representation, input_size, self.num_classes)

        print(dnn_representation)

        self.model.cuda()
        self.model.half()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        criterion = nn.CrossEntropyLoss().cuda()
        criterion.half()

        if load_weights:
            state_dict = torch.load(self.temp_weights_path + "model_weights_during_deepening.pth")
            new_state_dict = OrderedDict()

            for key, value in state_dict.items():
                if key != 'linear.weight' and key != 'linear.bias':
                    new_state_dict[key] = value

            self.model.load_state_dict(new_state_dict, strict=False)

        epochs_no_improve = 0
        epoch = 0

        best_val_loss = 0
        best_val_acc = 0
        best_model = None

        while epochs_no_improve < 5:
            epoch += 1
            print("\tEpoch [{}]:".format(epoch))

            # phase --> 'training'
            train_loss, train_acc = self.one_iter_train_non_medical(train_loader, iteration,
                                                        optimizer,
                                                        criterion,
                                                        phase='training')

            print("\t\tTraining Loss: {:.4f}, Training Acc: {:.4f}".format(train_loss, train_acc))

            if epoch % 1 == 0:
                # phase --> 'validation'
                val_loss, val_acc = self.one_iter_train_non_medical(validation_loader, iteration, optimizer, criterion,
                                                                phase='test')

                print("\t\tValidation Loss: {:.4f}, Validation Acc: {:.4f}".format(val_loss, val_acc))

                if val_acc > best_val_acc:
                    epochs_no_improve = 0
                    best_val_acc = val_acc.item()
                    best_val_loss = val_loss.item()
                    best_model = deepcopy(self.model)
                else:
                    epochs_no_improve += 1

        return best_model, best_val_acc, best_val_loss

    def one_iter_train_non_medical(self, data_loader, iteration, optimizer, criterion, phase='training'):
        running_loss = 0.0
        running_corrects = 0.0
        data_size = 0

        for _, (images, labels) in enumerate(data_loader):
            images = images.cuda().half()
            labels = labels.cuda()
            data_size += len(labels)

            if phase == 'training':
                self.model.train(True)
                # Forward pass
                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # Backward and optimize
                loss.backward()
                optimizer.step()
            else:
                self.model.eval()
                with torch.no_grad():
                    images = Variable(images)
                    labels = Variable(labels)

                    optimizer.zero_grad()

                    outputs = self.model(images)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

            running_loss += loss * labels.size(0)
            running_corrects += torch.sum(preds == labels.data)

        loss = running_loss / data_size
        acc = running_corrects / data_size

        return loss, acc

    def test_model(self, test_loader):
        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            criterion = nn.BCEWithLogitsLoss().cuda()
            criterion.half()
            test_loss, test_acc, test_AUC, _, _, test_sensitivity, test_specificity = \
                self.one_iter_train_medical(test_loader, 0, optimizer, criterion, phase="test")

            self.deepening_results = tabulate(
                [[str(test_loss.item()), str(test_acc), str(test_AUC), str(test_sensitivity), str(test_specificity),
                  str(self.flops)]],
                headers=["Test Loss", "Test Acc", "Test AUC", "Test Sensitivity", "Test Specificity", "FLOPs"],
                tablefmt="grid")
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            criterion = nn.CrossEntropyLoss().cuda()
            criterion.half()
            test_loss, test_acc = self.one_iter_train_non_medical(test_loader, 0, optimizer, criterion, phase="test")

            self.deepening_results = tabulate(
                [[str(test_loss.item()), str(test_acc), str(self.flops)]],
                headers=["Test Loss", "Test Acc", "FLOPs"],
                tablefmt="grid")
