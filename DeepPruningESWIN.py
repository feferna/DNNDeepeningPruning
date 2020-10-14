import matplotlib.pyplot as plt

from DNNPruningPopulation import *


def bitwise_mutation(individual_gene, mutation_p):
    for i in range(len(individual_gene)):
        if np.random.rand() <= mutation_p:
            individual_gene[i] = not individual_gene[i]

    return individual_gene


class DeepPruningESWIN:
    def __init__(self, alg_parameters, model_representation, model_weights_path, train_loader, valid_loader,
                 test_loader):
        self.parameters = alg_parameters

        self.model_representation = model_representation
        self.knee_preference = None
        self.heavy = None
        self.light = None

        # Initialize population
        self.pop = Population(alg_parameters, model_representation, model_weights_path, train_loader, valid_loader,
                              test_loader)

    def fit(self):
        for g in range(self.parameters.max_gen):
            print("Generation: " + str(g))

            # Non-dominated set selection
            print("\tNon-domination Selection...")
            if g == 0:
                F = []
            else:
                F = self.non_dominated_selection()

            # MMD Knee and Boundary Selection
            print("\tKnee and boundary individuals selection...")
            Q = self.knee_boundary_selection(g, F)

            print("\tOffspring generation...")
            self.offspring_generation(Q, g)            
        
        self.knee_preference, self.heavy, self.light = get_knee_boundary_solution(Q)

    def non_dominated_selection(self):
        IsDominated = [False]*len(self.pop.individual)
        F = []
        for i in range(len(self.pop.individual)):
            if not IsDominated[i]:
                for j in range(len(self.pop.individual)):
                    if i != j:
                        Pi = self.pop.individual[i].fitness
                        Pj = self.pop.individual[j].fitness
                        if Pj[0] <= Pi[0] and Pj[1] <= Pi[1]:
                            if Pj[0] < Pi[0] or Pj[1] < Pi[1]:
                                IsDominated[i] = True
                                break
                        elif Pi[0] <= Pj[0] and Pi[1] <= Pj[1]:
                            if Pi[0] < Pj[0] or Pi[1] < Pj[1]:
                                IsDominated[j] = True

        for i in range(len(IsDominated)):
            if not IsDominated[i]:
                F.append(deepcopy(self.pop.individual[i]))

        return F

    def knee_boundary_selection(self, gen, F):
        if gen == 0:
            Q = []
            for i in range(3):
                Q.append(deepcopy(self.pop.individual[i]))
                Q[-1].knee_preference = False
                Q[-1].boundary_heavy = False
                Q[-1].boundary_light = False
                if i == 0:
                    Q[0].evaluate()
                else:
                    Q[i].fitness = deepcopy(Q[0].fitness)
        else:
            for i in range(len(F)):
                F[i].knee_preference = False
                F[i].boundary_heavy = False
                F[i].boundary_light = False

            Q = []

            # Get the objective values of all individuals in the front
            ind_objs = np.array([x.fitness for x in F])

            # Find the minimum and maximum in each objective
            f1_arr = np.argsort(ind_objs[:, 0])
            f2_arr = np.argsort(ind_objs[:, 1])

            f1_min = ind_objs[f1_arr[0], 0]
            f1_max = ind_objs[f1_arr[-1], 0]

            f2_min = ind_objs[f2_arr[0], 1]
            f2_max = ind_objs[f2_arr[-1], 1]

            # Add the boundary individual in f1
            F[f1_arr[0]].boundary_heavy = True
            F[f1_arr[0]].boundary_light = False
            F[f1_arr[0]].knee_preference = False
            print("\t\tBoundary Heavy: " + str(F[f1_arr[0]].fitness))

            Q.append(deepcopy(F[f1_arr[0]]))

            # Add the boundary individual in f2
            F[f2_arr[0]].boundary_heavy = False
            F[f2_arr[0]].boundary_light = True
            F[f2_arr[0]].knee_preference = False
            print("\t\tBoundary Light: " + str(F[f2_arr[0]].fitness))

            Q.append(deepcopy(F[f2_arr[0]]))

            # Rank objectives f1 --> Training error; f2 --> FLOPs
            r = [1, 2]
            phi = [2 - r[0] + 1, 2 - r[1] + 1]

            # Compute the weighting vector
            w = [(phi[0]/(phi[0] + phi[1])), (phi[1]/(phi[0] + phi[1]))]

            # Determine the ideal vector
            y_ideal = [(f1_min/(f1_max-f1_min)), (f2_min/(f2_max-f2_min))]

            ### WIN and MMD
            norm = np.zeros((len(F)))
            # distances = np.zeros((len(F)))

            for i in range(len(F)):
                if F[i].boundary_heavy or F[i].boundary_light:
                    norm[i] = 100.0
                else:
                    norm[i] = w[0]*((F[i].fitness[0]/(f1_max - f1_min)) - y_ideal[0]) +\
                              w[1]*((F[i].fitness[1]/(f2_max - f2_min)) - y_ideal[1])

            F[np.argmin(norm)].boundary_heavy = False
            F[np.argmin(norm)].boundary_light = False
            F[np.argmin(norm)].knee_preference = True
            Q.append(deepcopy(F[np.argmin(norm)]))
            print("\t\tPreferable Knee: " + str(F[np.argmin(norm)].fitness))

            print("\t\tWIN distances: " + str(norm))

            if gen == 1 or gen == 5 or gen == 10:
                self.plot_knee_boundaries(gen, Q)

        return Q         
    
    def offspring_generation(self, Q, gen):
        for i in range(self.parameters.pop_size):
            self.pop.individual[i] = None

        new_ind_counter = 0

        if gen != 0:
            for i in range(len(Q)):
                self.pop.individual[i] = deepcopy(Q[i])
                
            new_ind_counter = len(Q)

        if gen != 0:
            boundary_heavy = deepcopy(Q[0])
            boundary_light = deepcopy(Q[1])

        while new_ind_counter < self.parameters.pop_size:
            parent_idx = np.random.randint(len(Q))

            offspring = deepcopy(Q[parent_idx])
            offspring = self.mutation(offspring)

            offspring.evaluate()
            
            try:
                self.pop.individual[new_ind_counter] = deepcopy(offspring)
            except IndexError:
                print("New population is ready...")

            new_ind_counter += 1

    def mutation(self, ind, mutation_prob=None):
        if mutation_prob is not None:
            m_prob = mutation_prob
        else:
            m_prob = self.parameters.mutation_p

        ind.genes[0] = bitwise_mutation(ind.genes[0], m_prob)
        ind.genes[1] = bitwise_mutation(ind.genes[1], m_prob)

        return ind

    def plot_knee_boundaries(self, gen, Q):
        fig1 = plt.figure(1)

        if gen == 1:
            marker_color = "blue"
        elif gen == 5:
            marker_color = "green"
        elif gen == 10:
            marker_color = "red"

        children_obj = np.array([x.fitness for x in self.pop.individual])

        plt.scatter(x=children_obj[:, 0], y=children_obj[:, 1], label="Offspring - Generation " + str(gen),
                    marker="+", color=marker_color)

        for i in range(len(Q)):
            if Q[i].knee_preference:
                plt.scatter(x=Q[i].fitness[0], y=Q[i].fitness[1],
                            label="Preferable Knee - Generation " + str(gen), marker='o', color=marker_color)

            if Q[i].boundary_heavy:
                plt.scatter(x=Q[i].fitness[0], y=Q[i].fitness[1],
                            label="Boundary heavy - Generation " + str(gen), marker='s', color=marker_color)

            if Q[i].boundary_light:
                plt.scatter(x=Q[i].fitness[0], y=Q[i].fitness[1],
                            label="Boundary light - Generation " + str(gen), marker='D', color=marker_color)

        if self.parameters.dataset_name == "ISIC2016" or self.parameters.dataset_name == "ChestXRay":
            plt.xlabel("(1 - Validation AUC)")
            plt.ylabel("Number of Floating Operations")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            plt.xlabel("Validation Error")
            plt.ylabel("Number of Floating Operations")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        path_figure = "./dnn_pruning_results/" + self.parameters.dataset_name + "/graphs/"

        if not os.path.exists(path_figure):
            os.makedirs(path_figure)

        fig1.savefig(path_figure + "gen_" + str(gen) + ".svg", bbox_inches='tight')

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.clear()

        if gen == 1:
            marker_color = "blue"
        elif gen == 10:
            marker_color = "green"
        elif gen == 20:
            marker_color = "red"

        children_obj = np.array([x.fitness for x in self.pop.individual])

        ax2.scatter(x=children_obj[:, 0], y=children_obj[:, 1], label="Offspring",
                    marker="+", color=marker_color)

        for i in range(len(Q)):
            if Q[i].knee_preference:
                ax2.scatter(x=Q[i].fitness[0], y=Q[i].fitness[1],
                            label="Preferable Knee", marker='o', color=marker_color)

            if Q[i].boundary_heavy:
                ax2.scatter(x=Q[i].fitness[0], y=Q[i].fitness[1],
                            label="Boundary heavy", marker='s', color=marker_color)

            if Q[i].boundary_light:
                ax2.scatter(x=Q[i].fitness[0], y=Q[i].fitness[1],
                            label="Boundary light", marker='D', color=marker_color)

        if self.parameters.dataset_name == "ISIC2016" or self.parameters.dataset_name == "ChestXRay":
            ax2.set_title("Generation: " + str(gen))
            ax2.set_xlabel("(1 - Validation AUC)")
            ax2.set_ylabel("Number of Floating Operations")
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax2.set_title("Generation: " + str(gen))
            ax2.set_xlabel("Validation Error")
            ax2.set_ylabel("Number of Floating Operations")
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig2.savefig(path_figure + "only_gen_" + str(gen) + ".svg", bbox_inches='tight')
