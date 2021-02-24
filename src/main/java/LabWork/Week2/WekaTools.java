package LabWork.Week2;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class WekaTools {
    public static final String ARFF_FILES_DIR = "./src/main/java/LabWork/Files/";

    public static Instances loadClassificationData(String fullPath) throws IOException {
        FileReader reader = new FileReader(fullPath);
        Instances newInstances = new Instances(reader);
        newInstances.setClassIndex(newInstances.numAttributes() - 1);
//        System.out.println("Data successfully loaded");
//        System.out.println("\t* Number of Instances: " + newInstances.numInstances());
//        System.out.println("\t* Number of Attributes: " + newInstances.numAttributes());
//        System.out.println("\t* Class Attribute: " + newInstances.classAttribute());
        return newInstances;
    }

    public static Instances[] splitData(Instances all, double testProportion) {
        if (testProportion > 1 || testProportion < 0) {
            throw new IllegalArgumentException("The proportion of instances to be used as test " +
                    "data must be a decimal value between 0 and 1");
        }
        int testInstancesCount = (int) Math.round(testProportion * all.numInstances());
        Instances[] split = new Instances[2];
        all.randomize(new Random());

        split[0] = new Instances(all);
        split[1] = new Instances(all, 0);
        while (testInstancesCount > 0) {
            split[1].add(split[0].remove(0));
            testInstancesCount--;
        }
        return split;
    }

    public static int[] classifyInstances(Classifier c, Instances testData) throws Exception {
        int[] predictions = new int[testData.numInstances()];
        int index = 0;

        for (Instance instance : testData) {
            predictions[index++] = (int) c.classifyInstance(instance);
        }
        return predictions;
    }

    public static int[] getClassValues(Instances data) {
        int[] classValues = new int[data.numInstances()];
        int index = 0;

        for (Instance ins : data) {
            classValues[index++] = (int) ins.classValue();
        }
        return classValues;
    }


    public static double accuracy(Classifier c, Instances testData) throws Exception {
        double accuratePredictions = 0;
        for (Instance instance : testData) {
            if (instance.classValue() == c.classifyInstance(instance)) {
                accuratePredictions++;
            }
        }
        return accuratePredictions / testData.numInstances();
    }

    public static double[] classDistribution(Instances data) {
        int totalInstances = data.numInstances();
        int[] nominalCounts = data.attributeStats(data.classIndex()).nominalCounts;
        double[] distribution = new double[nominalCounts.length];

        for (int i = 0; i < nominalCounts.length; i++) {
            distribution[i] = (double) nominalCounts[i] / totalInstances;
        }
        return distribution;
    }

    /**
     * numClasses required to build confusion matrix because a given class may not appear in either
     * the predicted or actual arrays but should still be represented.
     *
     * TP = True Positive  (predicted positive and was actually positive)
     * TN = True Negative  (predicted negative and was actually negative)
     * FN = False Negative (predicted negative but was actually positive)
     * FP = False Positive (predicted positive but was actually negative)
     *
     *                Actual
     *                0      1      2
     * Predicted   0  TP     FP
     *             1  FN     TN
     *             2
     */
    public static int[][] confusionMatrix(int[] predicted, int[] actual, int numClasses) {
        int[][] confusionMatrix = new int[numClasses][numClasses];
        for (int i = 0; i < actual.length; i++) {
            int classVal = actual[i];
            int predictedVal = predicted[i];
            confusionMatrix[predictedVal][classVal]++;
        }
        return confusionMatrix;
    }

    public static void printConfusionMatrix(int[][] confusionMatrix) {
        for (int i = 2; i < confusionMatrix.length; i++) {
            System.out.print("\t");
        }
        System.out.println("\t\t\t\tActual");
        for (int i = 0; i < confusionMatrix.length; i++) {
            if (i == confusionMatrix.length / 2) {
                System.out.print("Predicted\t");
            } else {
                System.out.print("\t\t\t");
            }
            for (int frequency : confusionMatrix[i]) {
                System.out.print(frequency + "\t|\t");
            }
            System.out.println("\n");
        }
    }
}
