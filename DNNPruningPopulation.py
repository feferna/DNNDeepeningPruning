from DNNPruningIndividual import *
from copy import deepcopy


def get_knee_boundary_solution(Q):
    for i in range(len(Q)):
        if Q[i].knee_preference:
            print("Saving Preferable Knee Solution: ")
            Q[i].save_model()

        if Q[i].boundary_heavy:
            print("Saving Boundary Heavy Solution: ")
            Q[i].save_model()

        if Q[i].boundary_light:
            print("Saving Boundary Light Solution: ")
            Q[i].save_model()

    for i in range(len(Q)):
        if Q[i].knee_preference:
            print("Retraining Preferable Knee Solution: ")
            # Save models in a file
            Q[i].retrain()
            print(Q[i].fitness)
            knee = deepcopy(Q[i])

        if Q[i].boundary_heavy:
            print("Retraining Boundary Heavy Solution: ")
            Q[i].retrain()
            print(Q[i].fitness)
            heavy = deepcopy(Q[i])

        if Q[i].boundary_light:
            print("Retraining Boundary Light Solution: ")
            Q[i].retrain()
            print(Q[i].fitness)
            light = deepcopy(Q[i])

    return knee, heavy, light


class Population:
    def __init__(self, alg_parameters, model_representation, model_weights_path, train_loader, valid_loader,
                 test_loader):
        self.model_representation = model_representation

        self.pop_size = alg_parameters.pop_size

        self.individual = [Individual(alg_parameters, model_representation, model_weights_path, train_loader,
                                      valid_loader, test_loader)]

        for _ in range(1, self.pop_size):
            self.individual.extend([Individual(alg_parameters, model_representation, model_weights_path, train_loader,
                                               valid_loader, test_loader)])

    def extend_population(self, offspring):
        self.individual.extend(offspring.individual)
