package LabWork.Week2;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.EntropySplitCrit;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instances;

import java.io.IOException;
import java.text.DecimalFormat;

public class DecisionTreesLab {

    public static void main(String[] args) throws Exception {
        Instances trainingData = WekaTools.loadClassificationData(WekaTools.ARFF_FILES_DIR +
                "UCIContinuous/breast-cancer-wisc-diag/breast-cancer-wisc-diag_TRAIN.arff");
        J48 j48 = new J48();
        j48.buildClassifier(trainingData);
        System.out.println("Default");
        System.out.println(j48 + "\n");

        j48.setBinarySplits(true);
        j48.buildClassifier(trainingData);
        System.out.println("Binary Splits = true");
        System.out.println(j48 + "\n");

        j48.setBinarySplits(false);
        j48.setReducedErrorPruning(true);
        j48.buildClassifier(trainingData);
        System.out.println(j48 + "\n");

        DecimalFormat df = new DecimalFormat("##.###");
        // {sunny -> 3 loss, 2 win}  {overcast -> 0 loss, 4 win}  {rain -> 2 loss, 3 win}
        double[][] outlook = new double[][] {{3, 2},{0,4},{2, 3}};
        Distribution d = new Distribution(outlook);
        InfoGainSplitCrit igsc = new InfoGainSplitCrit();
        EntropySplitCrit esc = new EntropySplitCrit();
        // 1 / value to find info gain split because method returns reciprocal
        System.out.println(df.format(1 / igsc.splitCritValue(d)));
        System.out.println(df.format(1 / esc.splitCritValue(d)));
    }
}
