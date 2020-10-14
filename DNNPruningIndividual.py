import sys
sys.path.append("./Utils/")

import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile

import numpy as np

import pruning_utils

from BuildModelResidual import CreateNetworkResidual

import os

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from torch.autograd import Variable

from collections import OrderedDict


def create_layer_list(model_representation):
    middle_filters = {}
    block_filters = {}

    for i in range(len(model_representation)):
        block_filters['residual' + str(i)] = model_representation[i]["Num. Feature Maps"]
        init_string = 'residual' + str(i) + "."

        for j in range(model_representation[i]["Num. Residual Layers"]):
            string = init_string + str(j) + ".conv1"

            middle_filters[string] = model_representation[i]["Num. Feature Maps"]

    return [middle_filters, block_filters]


class Individual:
    def __init__(self, alg_parameters, model_representation, model_weights_path, train_loader, validation_loader,
                 test_loader):
        self.dataset_name = alg_parameters.dataset_name
        self.input_size = alg_parameters.input_size

        self.model_representation = model_representation
        self.model_weights_path = model_weights_path

        # Get the number of FLOPS from the original model
        original_model = self.load_initial_model()
        self.original_flops, _ = profile(original_model, self.input_size)
        del original_model

        # self.train_loader_sample = train_loader_sample
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.partial_train_epochs = alg_parameters.partial_train_epochs
        self.full_train_epochs = alg_parameters.full_train_epochs
        self.learning_rate = alg_parameters.learning_rate

        self.knee_preference = False
        self.knee = False
        self.boundary_heavy = False
        self.boundary_light = False

        self.filters = create_layer_list(model_representation)
        self.genes_sizes = [sum(self.filters[0].values()), sum(self.filters[1].values())]
        
        # Gene is a list where the first position encodes the middle layers,
        #                and the second position encodes the residual blocks
        self.genes = [np.ones((self.genes_sizes[0],), dtype=np.bool), np.ones((self.genes_sizes[1],), dtype=np.bool)]

        self.fitness = [100, np.inf]
        self.decrease_flops = 0

    def load_initial_model(self):
        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            number_classes = 1
        elif self.dataset_name == "CIFAR10":
            number_classes = 10
        else:
            number_classes = 100

        model = CreateNetworkResidual(self.model_representation, self.input_size, num_classes=number_classes)

        model.cuda()

        state_dict = torch.load(self.model_weights_path)
        model.load_state_dict(state_dict, strict=False)

        return model

    def decode_pruning(self, model):
        model = pruning_utils.residual_pruning(model, self.genes, self.filters)

        return model

    def evaluate(self):
        # Copy original model
        model = self.load_initial_model()
        model = self.decode_pruning(model)

        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            criterion = nn.BCEWithLogitsLoss().cuda().half()
        else:
            criterion = nn.CrossEntropyLoss().cuda().half()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

        model.cuda()
        model.half()
        model.train()

        print("\t\tIndividual's partial retraining... Epoch:", end=" ")
        for epoch in range(self.partial_train_epochs):
            print(str(epoch), end=" ")
            running_loss = 0.0
            data_size = 0

            for i, (images, labels) in enumerate(self.train_loader):
                images = images.cuda().half()

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    labels = labels.cuda().half()
                    labels = labels.unsqueeze(1)
                else:
                    labels = labels.cuda()

                data_size += len(labels)

                # compute output
                optimizer.zero_grad()
                outputs = model(images)

                out_probs = torch.sigmoid(outputs)
                
                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)

                # Backward and optimize
                loss.backward()
                optimizer.step()

            train_loss = running_loss / data_size

        running_loss = 0.0
        running_corrects = 0

        data_size = 0

        y_pred = []
        y_true = []

        print("")

        for i, (images, labels) in enumerate(self.validation_loader):
            images = images.cuda().half()

            if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                labels = labels.cuda().half()
                labels = labels.unsqueeze(1)
                y_true.extend(labels.cpu().numpy())
            else:
                labels = labels.cuda()

            data_size += len(labels)

            model.eval()

            with torch.no_grad():
                images = Variable(images)
                labels = Variable(labels)

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    labels = labels.unsqueeze(1)

                # compute output
                optimizer.zero_grad()
                outputs = model(images)

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    y_pred.extend(outputs.squeeze().cpu().numpy())

                    out_probs = torch.sigmoid(outputs)
                else:
                    _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

            if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                running_loss += loss.item() * labels.size(0)
            else:
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            valid_loss = running_loss / data_size
        else:
            valid_loss = running_loss / data_size
            valid_acc = running_corrects / data_size

        flops, _ = profile(model.float(), input_size=self.input_size)

        self.decrease_flops = ((self.original_flops-flops)/self.original_flops)*100

        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            valid_auc = auc(fpr, tpr)

            self.fitness = [(1 - valid_auc), flops]
        else:
            self.fitness = [(1 - valid_acc), flops]

        print("\t\tIndividual's Validation Error: " + str(self.fitness[0]) +
              " Individual's # FLOPS: " + str(self.fitness[1]))

    def save_model(self):
        model = self.load_initial_model()
        model = self.decode_pruning(model)

        # Save model in a file
        path_log = "./dnn_pruning_results/" + self.dataset_name + "/saved_models_before_retraining/"

        if not os.path.exists(path_log):
            os.makedirs(path_log)

        if self.knee_preference:
            torch.save(model, path_log + "preferable_knee_model.pth")
        elif self.boundary_heavy:
            torch.save(model, path_log + "heavy_model.pth")
        elif self.boundary_light:
            torch.save(model, path_log + "light_model.pth")

    def retrain(self):
        model = self.load_initial_model()
        model = self.decode_pruning(model)

        model.cuda()
        model.half()

        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            criterion = nn.BCEWithLogitsLoss().cuda().half()
        else:
            criterion = nn.CrossEntropyLoss().cuda().half()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)

        running_loss = 0.0
        data_size = 0

        for epoch in range(self.full_train_epochs):
            print("Epoch [{}/{}]".format(epoch + 1, self.full_train_epochs))
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.cuda().half()

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    labels = labels.cuda().half()
                    labels = labels.unsqueeze(1)
                else:
                    labels = labels.cuda()

                data_size += len(labels)

                # compute output
                optimizer.zero_grad()
                outputs = model(images)

                out_probs = torch.sigmoid(outputs)

                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)

                # Backward and optimize
                loss.backward()
                optimizer.step()

            train_loss = running_loss / data_size

            print("\tTrain Loss: {:.4f}".format(train_loss))

            # Validation
            running_loss = 0.0
            running_corrects = 0

            data_size = 0

            y_pred = []
            y_true = []

            for i, (images, labels) in enumerate(self.validation_loader):
                images = images.cuda().half()

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    labels = labels.cuda().half()
                    labels = labels.unsqueeze(1)
                    y_true.extend(labels.cpu().numpy())
                else:
                    labels = labels.cuda()

                data_size += len(labels)

                model.eval()

                with torch.no_grad():
                    images = Variable(images)
                    labels = Variable(labels)

                    if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                        labels = labels.unsqueeze(1)

                    # compute output
                    optimizer.zero_grad()
                    outputs = model(images)

                    if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                        y_pred.extend(outputs.squeeze().cpu().numpy())

                        out_probs = torch.sigmoid(outputs)
                    else:
                        _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    running_loss += loss.item() * labels.size(0)
                else:
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

            if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                valid_loss = running_loss / data_size

                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                best_threshold_idx = np.argmax(tpr * (1 - fpr))
                valid_sensitivity = tpr[best_threshold_idx]
                valid_specificity = 1 - fpr[best_threshold_idx]
                decision_threshold = np.array([0.5])

                preds = np.where(y_pred >= decision_threshold, 1.0, 0.0)
                corrects = np.sum(preds == y_true)
                valid_acc = corrects / len(preds)

                valid_auc = auc(fpr, tpr)

                print("\tValidation Loss: {:.4f}".format(valid_loss))
                print("\tValidation Acc: {:.4f}".format(valid_acc))
                print("\tValidation AUC: {:.4f}".format(valid_auc))
                print("\tValidation Sensitivity: {:.4f}".format(valid_sensitivity))
                print("\tValidation Specificity: {:.4f}".format(valid_specificity))
            else:
                valid_loss = running_loss / data_size
                valid_acc = running_corrects / data_size

                print("\tValidation Loss: {:.4f}".format(valid_loss))
                print("\tValidation Acc: {:.4f}".format(valid_acc))

        # Test
        running_loss = 0.0
        running_corrects = 0.0
        data_size = 0

        y_pred = []
        y_true = []

        for i, (images, labels) in enumerate(self.test_loader):
            images = images.cuda().half()

            if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                labels = labels.cuda().half()
                labels = labels.unsqueeze(1)
                y_true.extend(labels.cpu().numpy())
            else:
                labels = labels.cuda()

            data_size += len(labels)

            model.eval()

            with torch.no_grad():
                images = Variable(images)
                labels = Variable(labels)

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    labels = labels.unsqueeze(1)

                # compute output
                optimizer.zero_grad()
                outputs = model(images)

                if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                    y_pred.extend(outputs.squeeze().cpu().numpy())

                    out_probs = torch.sigmoid(outputs)
                else:
                     _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

            if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
                running_loss += loss.item() * labels.size(0)
            else:
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            test_loss = running_loss / data_size

            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            best_threshold_idx = np.argmax(tpr * (1 - fpr))
            test_sensitivity = tpr[best_threshold_idx]
            test_specificity = 1 - fpr[best_threshold_idx]
            decision_threshold = np.array([0.5])

            preds = np.where(y_pred >= decision_threshold, 1.0, 0.0)
            corrects = np.sum(preds == y_true)
            test_acc = corrects / len(preds)

            test_auc = auc(fpr, tpr)

            print("\tTest Loss: {:.4f}".format(test_loss))
            print("\tTest Acc: {:.4f}".format(test_acc))
            print("\tTest AUC: {:.4f}".format(test_auc))
            print("\tTest Sensitivity: {:.4f}".format(test_sensitivity))
            print("\tTest Specificity: {:.4f}".format(test_specificity))
        else:
            test_loss = running_loss / data_size
            test_acc = running_corrects / data_size

            print("\tTest Loss: {:.4f}".format(test_loss))
            print("\tTest Acc: {:.4f}".format(test_acc))

        flops, _ = profile(model.float(), input_size=self.input_size)

        if self.dataset_name == "ISIC2016" or self.dataset_name == "ChestXRay":
            self.fitness = [test_acc, test_auc, test_sensitivity, test_specificity, flops]
        else:
            self.fitness = [test_acc, flops]

        # Save model in a file
        path_log = "./dnn_pruning_results/" + self.dataset_name + "/saved_models_after_retraining/"

        if not os.path.exists(path_log):
            os.makedirs(path_log)

        if self.knee_preference:
            torch.save(model, path_log + "preferable_knee_model.pth")
        elif self.boundary_heavy:
            torch.save(model, path_log + "heavy_model.pth")
        elif self.boundary_light:
            torch.save(model, path_log + "light_model.pth")
