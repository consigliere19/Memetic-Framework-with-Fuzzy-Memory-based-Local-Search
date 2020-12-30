####################################################################################################
# This file contains the code for the general framework of (MF)2LS
# For applications in different classes of problems, some parts of this code must be modified,
# which is highlighted in the comments below. Example code for Feature Selection is given.
# The required portions can be uncommented in order to run the program.
####################################################################################################


import numpy as np
import pandas as pd
import random
import math, time, sys, os
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest

####################################################################################################
# Define the local search procedure


def localSearch(agent, dvalue, feature_score):
    for i in range(len(agent)):
        if dvalue > random.random():
            if feature_score[i] > random.random():
                agent[i] = 1
            else:
                agent[i] = 0
    return agent


####################################################################################################
# Here the population is initialized


def initialize(popSize, dim):
    population = np.zeros((popSize, dim))
    minn = 1
    maxx = math.floor(0.8 * dim)
    if maxx < minn:
        minn = maxx

    for i in range(popSize):
        random.seed(i ** 3 + 10 + time.time())
        no = random.randint(minn, maxx)
        if no == 0:
            no = 1
        random.seed(time.time() + 100)
        pos = random.sample(range(0, dim - 1), no)
        for j in pos:
            population[i][j] = 1

    return population


####################################################################################################
# Here the fitness function is defined. This will be different for different classes of problems.
# The fitness function for Feature Selection is given below, as an example

# def fitness(solution, trainX, trainy, testX, testy):
#     cols = np.flatnonzero(solution)
#     val = 1
#     if np.shape(cols)[0] == 0:
#         return val
#     clf = KNeighborsClassifier(n_neighbors=5)
#     train_data = trainX[:, cols]
#     test_data = testX[:, cols]
#     clf.fit(train_data, trainy)
#     error = 1 - clf.score(test_data, testy)

#     featureRatio = solution.sum() / np.shape(solution)[0]
#     val = omega * error + (1 - omega) * featureRatio
#     return val


# def allfit(population, trainX, trainy, testX, testy):
#     x = np.shape(population)[0]
#     acc = np.zeros(x)
#     for i in range(x):
#         acc[i] = fitness(population[i], trainX, trainy, testX, testy)
#     return acc


####################################################################################################
# Roulette selection of parents for crossover


def selectParentRoulette(popSize, fitnList):
    fitnList = np.array(fitnList)
    fitnList = 1 - fitnList / fitnList.sum()
    random.seed(time.time() + 19)
    val = random.uniform(0, fitnList.sum())
    for i in range(popSize):
        if val <= fitnList[i]:
            return i
        val -= fitnList[i]
    return -1


####################################################################################################
# Main algorithm


def geneticAlgo(dataset, popSize, maxIter, randomstate):
    df = pd.read_csv(dataset)
    (a, b) = np.shape(df)
    data = df.values[:, 0 : b - 1]
    label = df.values[:, b - 1]

    ####################################################################################################
    # Here the initial ranking of the features is done. Again, this will be different for different classes of problems.

    # For example, a ranking method based on the ensemble of two filters for FS is shown below
    # ensemble_score1 = mutual_info_classif(data, label)
    # ensemble_score2 = SelectKBest(k="all").fit(data, label).scores_
    # ensemble_score = (ensemble_score1 + ensemble_score2) / 2
    # norm = np.linalg.norm(ensemble_score)
    # ensemble_score = ensemble_score / norm

    dimension = np.shape(data)[1]
    freq = np.zeros((dimension))

    cross = 5
    test_size = 1 / cross
    trainX, testX, trainy, testy = train_test_split(
        data, label, stratify=label, test_size=test_size, random_state=randomstate
    )

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX, trainy)
    val = clf.score(testX, testy)
    whole_accuracy = val

    x_axis = []
    y_axis = []
    population = initialize(popSize, dimension)
    fitList = allfit(population, trainX, trainy, testX, testy)

    fitindex = np.argsort(fitList)

    GBESTSOL = population[fitindex[0]].copy()
    GBESTFIT = fitList[fitindex[0]]

    start_time = datetime.now()

    feature_score = np.zeros((dimension))

    for currIter in range(maxIter):

        newpop = np.zeros((popSize, dimension))
        fitList = allfit(population, trainX, trainy, testX, testy)
        arr1inds = fitList.argsort()
        population = population[arr1inds]
        fitList = fitList[arr1inds]

        bestInx = np.argmin(fitList)
        fitBest = min(fitList)

        if fitBest < GBESTFIT:
            GBESTSOL = population[bestInx].copy()
            GBESTFIT = fitBest

        for selectioncount in range(int(popSize / 2)):
            parent1 = selectParentRoulette(popSize, fitList)
            parent2 = parent1
            while parent2 == parent1:
                random.seed(time.time())
                parent2 = selectParentRoulette(popSize, fitList)

            parent1 = population[parent1].copy()
            parent2 = population[parent2].copy()

            ####################################################################################################
            # crossover between parent1 and parent2

            child1 = parent1.copy()
            child2 = parent2.copy()
            for i in range(dimension):
                random.seed(time.time())
                if random.uniform(0, 1) < crossoverprob:
                    child1[i] = parent2[i]
                    child2[i] = parent1[i]
            i = selectioncount
            j = int(i + (popSize / 2))
            newpop[i] = child1.copy()
            newpop[j] = child2.copy()

        ####################################################################################################
        # mutation

        mutationprob = muprobmin + (muprobmax - muprobmin) * (currIter / maxIter)
        for index in range(popSize):
            for i in range(dimension):
                random.seed(time.time() + dimension + popSize)
                if random.uniform(0, 1) < mutationprob:
                    newpop[index][i] = 1 - newpop[index][i]

        ####################################################################################################
        # Feature scores are calculated according to the initial ranking and the frequency of occurrences in memory.
        # An example code snippet is given below

        # const1 = 0.5
        # const2 = 0.5
        # const3 = 0.5
        # for index in range(dimension):
        #     freq[index] += np.sum(newpop.T[index])
        # freq_sum = np.sum(freq)
        # for index in range(dimension):
        #     feature_score[index] = (
        #         const2 * freq[index] / freq_sum + const3 * ensemble_score[index]
        #     )

        fitnesses = allfit(newpop, trainX, trainy, testX, testy)
        fitindex = np.argsort(fitnesses)
        newpop = newpop[fitindex]
        for index in range(popSize):
            dvalue = const1 * (1 + index) / popSize
            old_agent = newpop[index]
            ####################################################################################################
            # Fuzzy memory based local search
            newpop[index] = localSearch(newpop[index], dvalue, feature_score)
            if fitness(old_agent, trainX, trainy, testX, testy) < fitness(
                newpop[index], trainX, trainy, testX, testy
            ):
                newpop[index] = old_agent

        population = newpop.copy()

    cols = np.flatnonzero(GBESTSOL)
    val = 1
    if np.shape(cols)[0] == 0:
        return GBESTSOL
    clf = KNeighborsClassifier(n_neighbors=5)
    train_data = trainX[:, cols]
    test_data = testX[:, cols]
    clf.fit(train_data, trainy)
    val = clf.score(test_data, testy)
    print(val, GBESTSOL.sum())
    return GBESTSOL, val


popSize = 20
maxIter = 30
omega = 0.99
crossoverprob = 0.6
muprobmin = 0.01
muprobmax = 0.3

####################################################################################################
# Initialize list of datasets

# datasetlist = [
#     "BreastEW.csv",
#     "Tic-tac-toe.csv",
#     "HeartEW.csv",
#     "M-of-n.csv",
#     "Vote.csv",
#     "CongressEW.csv",
#     "Lymphography.csv",
#     "SpectEW.csv",
#     "Ionosphere.csv",
#     "KrVsKpEW.csv",
#     "Sonar.csv",
#     "WaveformEW.csv",
#     "PenglungEW.csv",
# ]


for datasetinx in range(len(datasetlist)):
    dataset = datasetlist[datasetinx]
    best_accuracy = -100
    best_no_features = 100
    best_answer = []
    accuList = []
    featList = []
    for count in range(15):

        answer, testAcc = geneticAlgo(dataset, popSize, maxIter, 4)

        accuList.append(testAcc)
        featList.append(answer.sum())
        if testAcc >= best_accuracy and answer.sum() < best_no_features:
            best_accuracy = testAcc
            best_no_features = answer.sum()
            best_answer = answer.copy()
        if testAcc > best_accuracy:
            best_accuracy = testAcc
            best_no_features = answer.sum()
            best_answer = answer.copy()

    print("Best: ", best_accuracy, best_no_features)
