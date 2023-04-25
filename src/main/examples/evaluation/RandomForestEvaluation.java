package examples.evaluation;

import java.util.Random;

import com.classifiers.MyRandomForest;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RandomForestEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource trainSource = new DataSource("./data/train_nominal.arff");
        Instances trainset = trainSource.getDataSet();
        trainset.setClassIndex(trainset.numAttributes() - 1);

        MyRandomForest model = new MyRandomForest();
        model.buildClassifier(trainset);

        DataSource testSource = new DataSource("./data/test_nominal.arff");
        Instances testset = testSource.getDataSet();
        testset.setClassIndex(testset.numAttributes() - 1);

        Evaluation evaluator = new Evaluation(trainset);
        evaluator.crossValidateModel(model, testset, 10, new Random(0));
        System.out.println(evaluator.toClassDetailsString());

        evaluator.evaluateModel(model, testset);
        System.out.println("Correctly Classified Instances: " + evaluator.correct());
        System.out.println("Incorrectly Classified Instances: " + evaluator.incorrect());
        System.out.println("Percent of Correctly Classified Instances: " + evaluator.pctCorrect());
        System.out.println("Percent of Incorrectly Classified Instances: " + evaluator.pctIncorrect());
    }
}
