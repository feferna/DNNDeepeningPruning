import sys
sys.path.append("./Utils/")

import pickle
import time
import os

from tabulate import tabulate

from DataLoader import data_loader
from DNNDeepening import DNNDeepening
from DeepPruningESWIN import *


class DeepeningParameters:
    def __init__(self, dataset, G_trial, input_size, temp_weights_path):
        self.dataset_name = dataset
        self.G_trial = G_trial
        self.input_size = input_size
        self.temp_weights_path = temp_weights_path


class PruningParameters:
    def __init__(self, dataset_name, input_size, pop_size, max_generation, mutation_p, partial_train_epochs,
                 full_train_epochs,
                 learning_rate):
        self.dataset_name = dataset_name
        self.input_size = input_size

        self.pop_size = pop_size
        self.max_gen = max_generation
        self.mutation_p = mutation_p

        self.partial_train_epochs = partial_train_epochs
        self.full_train_epochs = full_train_epochs
        self.learning_rate = learning_rate


def DNNDeepeningOnly(deepening_parameters, train_loader, validation_loader, test_loader):
    dnn = DNNDeepening(deepening_parameters)

    start_time = time.time()
    dnn.fit(train_loader, validation_loader)
    end_time = time.time()

    gpu_hours = ((end_time - start_time)/60)/60

    save_results_path = "./dnn_deepening_results/" + deepening_parameters.dataset_name + "/"

    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    print_gpu_hours = "Total of GPU Hours: {:.4f}".format(gpu_hours)

    print(print_gpu_hours)

    with open(save_results_path + "GPU_hours.txt", "w") as f:
        print(print_gpu_hours, file=f)

    print("Best model found: ")
    print(dnn.list_blocks)

    with open(save_results_path + "best_model_representation.txt", "w") as f:
        print(dnn.list_blocks, file=f)

    print("Test model: ")
    dnn.test_model(test_loader)

    print(dnn.deepening_results)

    with open(save_results_path + "deepening_results.txt", "w") as f:
        print(dnn.deepening_results, file=f)

    with open(save_results_path + "best_model_representation.data", "wb") as f:
        pickle.dump(dnn.list_blocks, f)


def DNNPruningOnly(pruning_parameters, model_weights_path, train_loader, valid_loader, test_loader):
    with open("./dnn_deepening_results/" + pruning_parameters.dataset_name + "/best_model_representation.data", "rb")\
            as f:
        model_representation = pickle.load(f)

    # Run the algorithm...
    pruning_es = DeepPruningESWIN(pruning_parameters, model_representation, model_weights_path, train_loader,
                                  valid_loader, test_loader)

    start_time = time.time()
    pruning_es.fit()
    end_time = time.time()

    gpu_hours = ((end_time - start_time) / 60) / 60

    save_results_path = "./dnn_pruning_results/" + pruning_parameters.dataset_name + "/"

    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)

    # Save final results
    if pruning_parameters.dataset_name == "ISIC2016" or pruning_parameters.dataset_name == "ChestXRay":
        output_results = tabulate(
            [["Preferable Knee Solution", str(pruning_es.knee_preference.fitness[0]), str(pruning_es.knee_preference.fitness[1]),
              str(pruning_es.knee_preference.fitness[2]), str(pruning_es.knee_preference.fitness[3]),
              str(pruning_es.knee_preference.fitness[4]),
              str(pruning_es.knee_preference.decrease_flops)],
             ["Boundary Heavy", str(pruning_es.heavy.fitness[0]), str(pruning_es.heavy.fitness[1]),
              str(pruning_es.heavy.fitness[2]), str(pruning_es.heavy.fitness[3]), str(pruning_es.heavy.fitness[4]),
              str(pruning_es.heavy.decrease_flops)],
             ["Boundary Light", str(pruning_es.light.fitness[0]), str(pruning_es.light.fitness[1]),
              str(pruning_es.light.fitness[2]), str(pruning_es.light.fitness[3]), str(pruning_es.light.fitness[4]),
              str(pruning_es.light.decrease_flops)]],
            ["Individual", "Test Acc", "Test AUC (%)", "Test Sensitivity", "Test Specificity", "FLOPs",
             "FLOPs decrease (%)"],
            tablefmt="grid")
    else:
        output_results = tabulate(
            [["Preferable Knee Solution", str(pruning_es.knee_preference.fitness[0]),
              str(pruning_es.knee_preference.fitness[1]),
              str(pruning_es.knee_preference.decrease_flops)],
             ["Boundary Heavy", str(pruning_es.heavy.fitness[0]), str(pruning_es.heavy.fitness[1]),
              str(pruning_es.heavy.decrease_flops)],
             ["Boundary Light", str(pruning_es.light.fitness[0]), str(pruning_es.light.fitness[1]),
              str(pruning_es.light.decrease_flops)]],
            ["Individual", "Test Acc", "FLOPs",
             "FLOPs decrease (%)"],
            tablefmt="grid")

    print(output_results)

    with open(save_results_path + "pruning_results.txt", "w") as f:
        print(output_results, file=f)

    print_gpu_hours = "Total of GPU Hours: {:.4f}".format(gpu_hours)

    print(print_gpu_hours)

    with open(save_results_path + "GPU_hours.txt", "w") as f:
        print(print_gpu_hours, file=f)


if __name__ == "__main__":
    # Algorithm Parameters:
    #   1. Algorithm Mode:
    #       ALG_MODE = 0 --> Run DNN deepening followed by DNN Pruning
    #       ALG_MODE = 1 --> Run DNN deepening only
    #       ALG_MODE = 2 --> Run DNN Pruning only
    ALG_MODE = 0

    #   2. Dataset Parameters:
    DATASET_ROOT = "./datasets"

    DATASET_NAME = "ISIC2016"
    # DATASET_NAME = "ChestXRay"
    # DATASET_NAME = "CIFAR10"
    # DATASET_NAME = "CIFAR100"

    BATCH_SIZE = 4

    if DATASET_NAME == "ISIC2016" or DATASET_NAME == "ChestXRay":
        INPUT_SIZE = (1, 3, 224, 224)
    else:
        INPUT_SIZE = (1, 3, 32, 32)

    #   3. DNN Deepening Parameters
    G_TRIAL = 10
    TEMP_WEIGHTS_PATH = "./dnn_deepening_results/" + DATASET_NAME + "/"
    BEST_MODEL_WEIGHTS_PATH = "./dnn_deepening_results/" + DATASET_NAME + "/model_weights_during_deepening.pth"

    #   4. DNN Pruning Parameters
    POP_SIZE = 30
    MAX_GENERATION = 11
    MUTATION_P = 0.2  # Probability of mutation

    PARTIAL_TRAIN_EPOCHS = 3
    FULL_TRAIN_EPOCHS = 200
    LEARNING_RATE = 0.0001

    # Load dataset
    TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER = data_loader(DATASET_ROOT, DATASET_NAME, batch_size=BATCH_SIZE)

    # Run the algorithm
    DEEPENING_PARAMETERS = DeepeningParameters(DATASET_NAME,
                                           G_TRIAL,
                                           INPUT_SIZE,
                                           TEMP_WEIGHTS_PATH)

    PRUNING_PARAMETERS = PruningParameters(DATASET_NAME,
                                           INPUT_SIZE,
                                           POP_SIZE,
                                           MAX_GENERATION,
                                           MUTATION_P,
                                           PARTIAL_TRAIN_EPOCHS,
                                           FULL_TRAIN_EPOCHS,
                                           LEARNING_RATE)

    if ALG_MODE == 0:
        DNNDeepeningOnly(DEEPENING_PARAMETERS, TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER)
        DNNPruningOnly(PRUNING_PARAMETERS, BEST_MODEL_WEIGHTS_PATH, TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER)

    elif ALG_MODE == 1:
        DNNDeepeningOnly(DEEPENING_PARAMETERS, TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER)

    elif ALG_MODE == 2:
        DNNPruningOnly(PRUNING_PARAMETERS, BEST_MODEL_WEIGHTS_PATH, TRAIN_LOADER, VALIDATION_LOADER, TEST_LOADER)
