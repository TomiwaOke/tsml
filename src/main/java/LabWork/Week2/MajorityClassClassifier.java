package LabWork.Week2;

import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

public class MajorityClassClassifier implements Classifier {
    private double majorityClass = -1;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        int[] nominalCounts = data.attributeStats(data.classIndex()).nominalCounts;
        System.out.println(Arrays.toString(nominalCounts));
        for (int i = 0; i < nominalCounts.length; i++) {
            if (majorityClass < nominalCounts[i]) {
                majorityClass = i;
            }
        }
        System.out.println("Majority Class ->  " + majorityClass + "\n\n");
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (majorityClass == -1) {
            throw new RuntimeException("Classifier has not been built");
        }
        return majorityClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    public static void main(String[] args) throws Exception {
        Instances trainingData = WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Arsenal_TRAIN.arff");
//        Instances testData = WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Arsenal_TEST.arff");

        Instances[] dataSplit = WekaTools.splitData(trainingData, 0.5);
        MajorityClassClassifier mcc = new MajorityClassClassifier();
        ZeroR zr = new ZeroR();
        mcc.buildClassifier(dataSplit[0]);
        zr.buildClassifier(dataSplit[0]);

        System.out.println("************** Custom Majority Class Classifier *****************");
        System.out.println("Class Values: " + Arrays.toString(WekaTools.getClassValues(dataSplit[1])));
        System.out.println("Predicted Values: " + Arrays.toString(WekaTools.classifyInstances(mcc, dataSplit[1])));
        WekaTools.printConfusionMatrix(WekaTools.confusionMatrix(
                WekaTools.classifyInstances(mcc, dataSplit[1]), // predicted
                WekaTools.getClassValues(dataSplit[1]), // actual
                dataSplit[1].numClasses()
        ));
        System.out.println(WekaTools.accuracy(mcc, dataSplit[1]) + " Accuracy\n\n");

        System.out.println("************** ZeroR Classifier *****************");
        System.out.println("Class Values: " + Arrays.toString(WekaTools.getClassValues(dataSplit[1])));
        System.out.println("Predicted Values: " + Arrays.toString(WekaTools.classifyInstances(zr, dataSplit[1])));
        WekaTools.printConfusionMatrix(WekaTools.confusionMatrix(
                WekaTools.classifyInstances(zr, dataSplit[1]), // predicted
                WekaTools.getClassValues(dataSplit[1]), // actual
                dataSplit[1].numClasses()
        ));
        System.out.println(WekaTools.accuracy(zr, dataSplit[1]) + " Accuracy\n\n");
    }
}
