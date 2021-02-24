package LabWork.Week3;

import LabWork.Week2.WekaTools;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.Arrays;

public class EnsemblesLab {

    public static void randomVsBagging(Instances trainData, Instances testData) throws Exception {
        Bagging bg = new Bagging();
        RandomForest rf = new RandomForest();
        rf.setNumTrees(500);
        bg.buildClassifier(trainData);
        rf.buildClassifier(trainData);

        int[] actualVals = WekaTools.getClassValues(testData);
        int[] bgClassify = WekaTools.classifyInstances(bg, testData);
        int[] rfClassify = WekaTools.classifyInstances(rf, testData);

        System.out.println("\nBagging Classifier Run:");
        System.out.println("\tAccuracy -> " + WekaTools.accuracy(bg, testData));
        System.out.println("\tConfusion Matrix -> \n");
        WekaTools.printConfusionMatrix(WekaTools
                .confusionMatrix(bgClassify, actualVals, testData.numClasses()));


        System.out.println("\nRandom Forest Classifier Run:");
        System.out.println("\tAccuracy -> " + WekaTools.accuracy(rf, testData));
        System.out.println("\tConfusion Matrix -> \n");
        WekaTools.printConfusionMatrix(WekaTools
                .confusionMatrix(rfClassify, actualVals, testData.numClasses()));
    }

    public static void randomVsJ48(Instances trainData, Instances testData) throws Exception {
        J48 j48 = new J48();
        RandomForest rf = new RandomForest();
        rf.setNumTrees(500);
        j48.buildClassifier(trainData);
        rf.buildClassifier(trainData);

        int[] actualVals = WekaTools.getClassValues(testData);
        int[] j48Classify = WekaTools.classifyInstances(j48, testData);
        int[] rfClassify = WekaTools.classifyInstances(rf, testData);

        System.out.println("\nJ48 Classifier Run:");
        System.out.println("\tAccuracy -> " + WekaTools.accuracy(j48, testData));
        System.out.println("\tConfusion Matrix -> \n");
        WekaTools.printConfusionMatrix(WekaTools
                .confusionMatrix(j48Classify, actualVals, testData.numClasses()));


        System.out.println("\nRandom Forest Classifier Run:");
        System.out.println("\tAccuracy -> " + WekaTools.accuracy(rf, testData));
        System.out.println("\tConfusion Matrix -> \n");
        WekaTools.printConfusionMatrix(WekaTools
                .confusionMatrix(rfClassify, actualVals, testData.numClasses()));

    }

    public static void randomVsTomiwa(Instances trainData, Instances testData) throws Exception {
        TomiwaEnsemble te = new TomiwaEnsemble();
        RandomForest rf = new RandomForest();
        te.buildClassifier(trainData);
        rf.buildClassifier(trainData);

        int[] actualVals = WekaTools.getClassValues(testData);
        int[] teClassify = WekaTools.classifyInstances(te, testData);
        int[] rfClassify = WekaTools.classifyInstances(rf, testData);

        System.out.println("\nTomiwa Ensemble Classifier Run:");
        System.out.println("\tAccuracy -> " + WekaTools.accuracy(te, testData));
        System.out.println("\tConfusion Matrix -> \n");
        WekaTools.printConfusionMatrix(WekaTools
                .confusionMatrix(teClassify, actualVals, testData.numClasses()));


        System.out.println("\nRandom Forest Classifier Run:");
        System.out.println("\tAccuracy -> " + WekaTools.accuracy(rf, testData));
        System.out.println("\tConfusion Matrix -> \n");
        WekaTools.printConfusionMatrix(WekaTools
                .confusionMatrix(rfClassify, actualVals, testData.numClasses()));

    }

    public static void main(String[] args) throws Exception {
        Instances trainingData = WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR +
                "UCIContinuous/breast-cancer-wisc-diag/breast-cancer-wisc-diag_TRAIN.arff");
        Instances testData = WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR +
                "UCIContinuous/breast-cancer-wisc-diag/breast-cancer-wisc-diag_TEST.arff");

        System.out.println("\n\n*********** Breast Cancer **************");
        randomVsTomiwa(trainingData, testData);

        System.out.println("\n\n*********** Aresnal Results **************");
        randomVsTomiwa(
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Arsenal_TRAIN.arff"),
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Arsenal_TEST.arff"));

        System.out.println("\n\n*********** Blood DataSet **************");
        randomVsTomiwa(
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "UCIContinuous/blood/blood_TRAIN.arff"),
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "UCIContinuous/blood/blood_TEST.arff"));

        System.out.println("\n\n*********** Glass DataSet **************");
        randomVsTomiwa(
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "UCIContinuous/glass/glass_TRAIN.arff"),
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "UCIContinuous/glass/glass_TEST.arff"));

        System.out.println("\n\n*********** Iris DataSet **************");
        randomVsTomiwa(
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "UCIContinuous/iris/iris_TRAIN.arff"),
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "UCIContinuous/iris/iris_TEST.arff"));

        System.out.println("\n\n*********** Adiac DataSet **************");
        randomVsTomiwa(
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Adiac/Adiac_TRAIN.arff"),
                WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR + "Adiac/Adiac_TEST.arff"));
    }
}
