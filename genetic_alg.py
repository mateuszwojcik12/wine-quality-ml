import random

from deap import base
from deap import creator
from deap import tools
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

pd.set_option('display.max_columns', None)
# df = pd.read_csv("data.csv")
# y = df.iloc[:, 3].values

df = pd.read_csv("data_GA.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1:].values

numberOfAtributtes = len(df.columns)

mms = MinMaxScaler()
df_norm = mms.fit_transform(df)

clf = KNeighborsClassifier()
# clf=DecisionTreeClassifier()
# clf=LogisticRegression()
scores = model_selection.cross_val_score(clf, df_norm, y, n_jobs=-1)
print(scores.mean())


def function_implementation(individual):
    part1 = (individual[0] + 2 * individual[1] - 7) ** 2
    part2 = (2 * individual[0] + individual[1] - 5) ** 2
    return part1 + part2,


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))
    return icls(genome)


def KNNParametersFeature(numberFeatures, icls):
    genome = list()
    # n_neighbors
    list_n_neighbors = [5, 3, 7, 9]
    genome.append(list_n_neighbors[random.randint(0, 3)])
    # weights=['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    genome.append(algorithm[random.randint(0, 3)])
    leaf_size = [30, 50, 70, 80]
    genome.append(leaf_size[random.randint(0, 3)])
    genome.append(2)
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def DecisionTreeClassifierParametersFeature(numberFeatures, icls):
    genome = list()
    # n_neighbors
    criterion = ['gini', 'entropy']
    genome.append(criterion[random.randint(0, 1)])
    splitter = ['best', 'random']
    genome.append(splitter[random.randint(0, 1)])
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def LogisticRegressionParametersFeature(numberFeatures, icls):
    genome = list()
    # n_neighbors
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def KNNParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    # dfSelectedFeatures=df.drop(df.columns[listColumnsToDrop], axis=1,
    # inplace=False)
    dfSelectedFeatures = df
    mms = MinMaxScaler()
    # print('*****')
    # print(dfSelectedFeatures)
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = KNeighborsClassifier(n_neighbors=individual[0], algorithm=individual[1], leaf_size=individual[2], p=2)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train].ravel())
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        cm22 = metrics.confusion_matrix(expected,
                                                  predicted).ravel()
        tn, fp, fn, tp = cm22[0],cm22[5],cm22[10],cm22[15]
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def DecisionTreeParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1,
                                 inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = DecisionTreeClassifier(criterion=individual[0], splitter=individual[1])
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,
                                                  predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def SVCParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    estimator = SVC(kernel=individual[0], C=individual[1], degree=individual[2], gamma=individual[3],
                    coef0=individual[4], random_state=101)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    result = ((resultSum / split),)
    return result


def LogisticRegressionFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1,
                                 inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)
    estimator = LogisticRegression()
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
        predicted = estimator.predict(df_norm[test])
        expected = y[test]
        tn, fp, fn, tp = metrics.confusion_matrix(expected,
                                                  predicted).ravel()
        result = (tp + tn) / (tp + fp + tn + fn)
        resultSum = resultSum + result
    return resultSum / split,


def mutationDecisionTreeClassifier(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        criterion = ['gini', 'entropy']
        individual[0] = criterion[random.randint(0, 1)]
    elif numberParamer == 1:
        splitter = ['best', 'random']
        individual[1] = splitter[random.randint(0, 1)]
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0


def mutationKNN(individual):
    numberParamer = random.randint(0, len(individual) - 1)

    if numberParamer == 0:
        list_n_neighbors = [5, 3, 7, 9]
        individual[0] = list_n_neighbors[random.randint(0, 3)]
    elif numberParamer == 1:
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        individual[1] = algorithm[random.randint(0, 3)]
    elif numberParamer == 2:
        leaf_size = [30, 50, 70, 80]
        individual[2] = leaf_size[random.randint(0, 3)]
    elif numberParamer == 3:
        individual[3] = 2
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0


def mutationLogisticRegression(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        criterion = ['gini', 'entropy']
        individual[0] = criterion[random.randint(0, 1)]
    elif numberParamer == 1:
        splitter = ['best', 'random']
        individual[1] = splitter[random.randint(0, 1)]
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0


if __name__ == '__main__':
    print('Minimalizcja = 1\n'
          'Selekcja: turniejowa=0, random=1, najlepszych osobników=2, najgorszych=3, ruletki=4')
    minimizeInput = input('Minimalizacja:')
    minimize = True if minimizeInput == '1' else False
    selectionType = input('Selekcja:')
    print('Krzyżowanie:\n'
          'Jednopunktowe:0, Dwupunktowe:1, Jednorodne:2')
    crossType = input("Krzyżowanie:")
    print('Mutacja:\n'
          'Gaussa=0, Zamiana indeksów=1, Zamiana bitów=2')
    mutType = input('Rodzaj mutacji: ')
    if minimize:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    else:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    alg = input("Wybierz algorytm 0-regresja logistyczna, 1-knn, 2-drzewo decyzyjne")
    if (alg == '0'):
        toolbox.register('individual', LogisticRegressionParametersFeature, numberOfAtributtes, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", LogisticRegressionFeatureFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationLogisticRegression)
    elif (alg == '1'):
        toolbox.register('individual', KNNParametersFeature, numberOfAtributtes, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", KNNParametersFeatureFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationKNN)
    else:
        toolbox.register('individual', DecisionTreeClassifierParametersFeature, numberOfAtributtes, creator.Individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", DecisionTreeParametersFeatureFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationDecisionTreeClassifier)

    if selectionType == '0':
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif selectionType == '1':
        toolbox.register("select", tools.selRandom)
    elif selectionType == '2':
        toolbox.register("select", tools.selBest)
    elif selectionType == '3':
        toolbox.register("select", tools.selWorst)
    else:
        toolbox.register("select", tools.selRoulette)

    if crossType == '0':
        toolbox.register("mate", tools.cxTwoPoint)
    elif crossType == '1':
        toolbox.register("mate", tools.cxOnePoint)
    elif crossType == '2':
        toolbox.register("mate", tools.cxUniform, indpb=0.5)

    sizePopulation = 20
    probabilityMutation = 0.2
    probabilityCrossover = 0.8
    numberIteration = 40

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    bests = []
    avg = []
    min_list = []
    max_list = []
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        g = 0
        numberElitism = 1
        while g < numberIteration:
            g = g + 1
            print("-- Generation %i --" % g)
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            listElitism = []
            for x in range(0, numberElitism):
                listElitism.append(tools.selBest(pop, 1)[0])
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if random.random() < probabilityCrossover:
                    toolbox.mate(child1, child2)
                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < probabilityMutation:
                    toolbox.mutate(mutant)
                del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            print(" Evaluated %i individuals" % len(invalid_ind))
            pop[:] = offspring + listElitism
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            print(" Min %s" % min(fits))
            print(" Max %s" % max(fits))
            print(" Avg %s" % mean)
            print(" Std %s" % std)
            best_ind = tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            bests.append(best_ind.fitness.values[0])
            avg.append(mean)
            min_list.append(min(fits))
            max_list.append(max(fits))
        #
        print("-- End of (successful) evolution --")

    iterations = []
    for i in range(0, len(bests)):
        iterations.append(i)
    import matplotlib.pyplot as plt

    plt.plot(iterations, bests)
    plt.xlabel("Iteracja")
    plt.ylabel("Najlepszy wynik")
    plt.title("Wykres wartości najlepszych wyników funkcji celu")
    plt.show()

    plt.plot(iterations, avg)
    plt.xlabel("Iteracja")
    plt.ylabel("Średni wynik")
    plt.title("Wykres średnich wartości wyników funkcji celu")
    plt.show()

    plt.plot(iterations, min_list)
    plt.xlabel("Iteracja")
    plt.ylabel("Najmniejszy wynik")
    plt.title("Wykres najmniejszych wartości wyników funkcji celu")
    plt.show()

    plt.plot(iterations, max_list)
    plt.xlabel("Iteracja")
    plt.ylabel("Największy wynik")
    plt.title("Wykres największych wartości wyników funkcji celu")
    plt.show()