package examples.evaluation;

import java.util.Random;

import com.classifiers.MyRandomForest;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RandomForestEvaluation {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("./data/out.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        MyRandomForest model = new MyRandomForest();
        model.buildClassifier(dataset);

        Evaluation evaluator = new Evaluation(dataset);
        evaluator.crossValidateModel(model, dataset, 10, new Random(0));
        System.out.println(evaluator.toSummaryString());
    }
}
