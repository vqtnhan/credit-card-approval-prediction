package examples.evaluation;

import java.util.Random;

import com.classifiers.MyMultipleLinearRegression;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MultipleLinearRegressionEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("./data/out.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        MyMultipleLinearRegression model = new MyMultipleLinearRegression();
        model.buildClassifier(dataset);

        Evaluation evaluator = new Evaluation(dataset);
        evaluator.crossValidateModel(model, dataset, 10, new Random(0));
        System.out.println(evaluator.toSummaryString());
    }
}
