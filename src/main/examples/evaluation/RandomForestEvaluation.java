package examples.evaluation;

import java.util.Random;

import com.classifiers.MyRandomForest;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RandomForestEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource trainSource = new DataSource("./data/train.arff");
        Instances trainset = trainSource.getDataSet();
        trainset.setClassIndex(trainset.numAttributes() - 1);

        MyRandomForest model = new MyRandomForest();
        model.buildClassifier(trainset);

        DataSource testSource = new DataSource("./data/test.arff");
        Instances testset = testSource.getDataSet();
        testset.setClassIndex(testset.numAttributes() - 1);

        Evaluation evaluator = new Evaluation(testset);
        evaluator.crossValidateModel(model, testset, 10, new Random(0));
        System.out.println(evaluator.toSummaryString());
    }
}
