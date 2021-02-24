package LabWork.Week3;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class TomiwaEnsemble extends AbstractClassifier {
    private ArrayList<Classifier> ensemble;
    private int numClassifiers;

    public TomiwaEnsemble() {
        numClassifiers = 500;
    }

    public void setNumClassifiers(int numClassifiers) {
        this.numClassifiers = numClassifiers;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        ensemble = new ArrayList<>();
        for (int i = 0; i < numClassifiers; i++) {
            data.randomize(new Random());
            Instances subset = new Instances(data, 0, data.numInstances() / 2);
            J48 newClassifier = new J48();
            newClassifier.buildClassifier(subset);
            ensemble.add(newClassifier);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        int[] classValues = new int[instance.numClasses()];
        for (Classifier c : ensemble) {
            classValues[(int) c.classifyInstance(instance)] += 1;
        }
        int majorityIndex = 0;
        for (int i = 1; i < classValues.length; i++) {
            if (classValues[majorityIndex] < classValues[i]) {
                majorityIndex = i;
            }
        }
        return majorityIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distributions = new double[instance.numClasses()];
        for (Classifier c : ensemble) {
            double[] singleDistribution = c.distributionForInstance(instance);
            for (int i = 0; i < distributions.length; i++) {
                distributions[i] += singleDistribution[i];
            }
        }
        double denominator = 0;
        for (double dist : distributions) {
            denominator += dist;
        }
        for (int i = 0; i < distributions.length; i++) {
            distributions[i] /= denominator;
        }
        return distributions;
    }
}
