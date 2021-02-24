package LabWork.Week1;

import LabWork.Week2.WekaTools;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.DecisionStump;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Map;

public class ClassifierTest {
    public static Instances loadData(String filePath) {
        try {
            FileReader reader = new FileReader(filePath);
            Instances newInstances = new Instances(reader);
            newInstances.setClassIndex(newInstances.numAttributes() - 1);
            return newInstances;
        } catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }
        return null;
    }

    public static void runClassifier(Classifier classifier, Instances trainingData, Instances testData) throws Exception {
        Map<Double, String> results = Map.of(2.0, "Win", 1.0, "Draw", 0.0, "Loss");
        classifier.buildClassifier(trainingData);

        if (testData != null) {
            for (Instance instance : testData) {
                System.out.println(instance.toString());
                System.out.println("\tClassifier's prediction: " + results
                        .get(classifier.classifyInstance(instance)));
                System.out.println("\tDistribution: " + Arrays
                        .toString(classifier.distributionForInstance(instance)));
            }
        }
        System.out.println("Accuracy = " + WekaTools.accuracy(classifier, testData));
    }

    public static void main(String[] args) throws Exception {
        Instances trainingData = WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Arsenal_TRAIN.arff");
        Instances testData = WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Arsenal_TEST.arff");
//
//        System.out.println(Arrays.toString(WekaTools.classDistribution(trainingData)));
//        System.out.println(WekaTools.splitData(trainingData, 0.5)[0]);


        NaiveBayes naiveBayes = new NaiveBayes();
        System.out.println("************************* Naive Bayes Run *************************");
        naiveBayes.buildClassifier(trainingData);
        System.out.println(Arrays.toString(WekaTools.getClassValues(testData)));
        System.out.println(Arrays.toString(WekaTools.classifyInstances(naiveBayes, testData)));
        WekaTools.printConfusionMatrix(WekaTools.confusionMatrix(
                WekaTools.classifyInstances(naiveBayes, testData), // predicted
                WekaTools.getClassValues(testData), // actual
                testData.numClasses()
        ));
        System.out.println(WekaTools.accuracy(naiveBayes, testData) + " Accuracy\n\n");



        String basePath = "src/main/java/experiments/data/tsc/";
        String dataset = "ItalyPowerDemand";
        Instances italyTrain = WekaTools.loadClassificationData(basePath + dataset + "/" + dataset + "_TRAIN.arff");
        Instances italyTest = WekaTools.loadClassificationData(basePath + dataset + "/" + dataset + "_TEST.arff");


        NaiveBayes naiveBayes2 = new NaiveBayes();
        System.out.println("************************* Naive Bayes Run [Italy] *************************");
        naiveBayes2.buildClassifier(italyTrain);
        System.out.println(Arrays.toString(WekaTools.getClassValues(italyTest)));
        System.out.println(Arrays.toString(WekaTools.classifyInstances(naiveBayes2, italyTest)));
        WekaTools.printConfusionMatrix(WekaTools.confusionMatrix(
                WekaTools.classifyInstances(naiveBayes2, italyTest), // predicted
                WekaTools.getClassValues(italyTest), // actual
                italyTest.numClasses()
        ));
        System.out.println(WekaTools.accuracy(naiveBayes2, italyTest) + " Accuracy\n\n");

        DecisionStump decisionStump = new DecisionStump();
        System.out.println("************************* Decision Stump Run [Italy] *************************");
        decisionStump.buildClassifier(italyTrain);
        System.out.println(Arrays.toString(WekaTools.getClassValues(italyTest)));
        System.out.println(Arrays.toString(WekaTools.classifyInstances(decisionStump, italyTest)));
        WekaTools.printConfusionMatrix(WekaTools.confusionMatrix(
                WekaTools.classifyInstances(decisionStump, italyTest), // predicted
                WekaTools.getClassValues(italyTest), // actual
                italyTest.numClasses()
        ));
        System.out.println(WekaTools.accuracy(decisionStump, italyTest) + " Accuracy\n\n");

//        IBk iBk = new IBk();
//        System.out.println("\n************************* IBk Run *************************");
//        runClassifier(iBk, trainingData, testData);
    }
}
