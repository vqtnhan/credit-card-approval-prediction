package examples.evaluation;

import java.util.Random;

import com.classifiers.MyMultipleLinearRegression;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MultipleLinearRegressionEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource trainSource = new DataSource("./data/train_numeric.arff");
        Instances trainset = trainSource.getDataSet();
        trainset.setClassIndex(trainset.numAttributes() - 1);

        MyMultipleLinearRegression model = new MyMultipleLinearRegression();
        model.buildClassifier(trainset);

        DataSource testSource = new DataSource("./data/test_numeric.arff");
        Instances testset = testSource.getDataSet();
        testset.setClassIndex(testset.numAttributes() - 1);

        Evaluation evaluator = new Evaluation(trainset);
        evaluator.crossValidateModel(model, testset, 10, new Random(0));
        System.out.println(evaluator.toSummaryString());

        double correctCounter = 0, incorrectCounter = 0;
        for (int i = 0; i < testset.numInstances(); ++i) {
            double label = testset.instance(i).classValue();
            double pred = model.classifyInstance(testset.instance(i));
            if (pred > 0.5)
                pred = 1;
            else
                pred = 0;

            if (pred == label)
                ++correctCounter;
            else
                ++incorrectCounter;
        }

        System.out.println("Correctly Classified Instances: " + correctCounter);
        System.out.println("Incorrectly Classified Instances: " + incorrectCounter);
        System.out.println("Percent of Correctly Classified Instances: " + correctCounter / testset.numInstances() * 100);
        System.out.println("Percent of Incorrectly Classified Instances: " + incorrectCounter / testset.numInstances() * 100);
    }
}
