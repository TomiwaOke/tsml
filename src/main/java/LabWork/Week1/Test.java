package LabWork.Week1;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileReader;
import java.util.Arrays;

public class Test {

    public static Instances loadData(String filePath) {
        try {
            FileReader reader = new FileReader(filePath);
            return new Instances(reader);
        } catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
        return null;
    }

    public static Instances removeAttribute(Instances ins, int[] removingIndices) throws Exception {
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(removingIndices);
        // removeFilter.setInvertSelection(true);
        // add above line when removing indices has the attribute indices to keep
        removeFilter.setInputFormat(ins);

        return Filter.useFilter(ins, removeFilter);
    }

    public static void printMetrics(Instances data) {
        System.out.println("Num Instances: " + data.numInstances());
        System.out.println("Num Attributes: " + data.numAttributes());
        double[] resultsArr = data.attributeToDoubleArray(3);
        System.out.println("Num Wins: " + Arrays.stream(resultsArr)
                .filter(value -> value == 2.0)
                .toArray()
                .length);
        System.out.println("Alternative :" + data.attributeStats(3).nominalCounts[2]);
        System.out.println("5th Instance: " + Arrays.toString(data.get(4).toDoubleArray()));
        System.out.println("Instances:");
        System.out.println(data.toString());

        try {
            System.out.println("Without Saka (Attribute[2]):");
            System.out.println(removeAttribute(data, new int[]{2}).toString());
        } catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
    }

    public static void main(String[] args) {
        String PATH = "./src/main/java/LabWork/Files/";
        Instances trainingData = loadData(PATH + "Arsenal_TRAIN.arff");
        if (trainingData != null) {
            System.out.println("******************************* Training Data ***************************************");
            printMetrics(trainingData);
        }

        Instances testData = loadData(PATH + "Arsenal_TEST.arff");
        if (testData != null) {
            System.out.println("\n*********************************** Test Data *************************************");
            printMetrics(testData);
        }
    }
}
