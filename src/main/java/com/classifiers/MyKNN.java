package com.classifiers;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class MyKNN extends AbstractClassifier {
    protected int m_K = 1;

    protected NearestNeighbourSearch linearNNSearch = new LinearNNSearch();

    public void buildClassifier(Instances trainingData) throws Exception {
        linearNNSearch.setInstances(trainingData);
    }

    public double[] distributionForInstance(Instance testInstance) throws Exception {
        linearNNSearch.addInstanceInfo(testInstance);

        Instances neighbours = linearNNSearch.kNearestNeighbours(testInstance, m_K);

        double[] dist = new double[testInstance.numClasses()];
        for (Instance neighbour : neighbours) {
            if (testInstance.classAttribute().isNominal()) {
                dist[(int) neighbour.classValue()] += 1.0 / neighbours.numInstances();
            } else {
                dist[0] += neighbour.classValue() / neighbours.numInstances();
            }
        }
        return dist;
    }
}
