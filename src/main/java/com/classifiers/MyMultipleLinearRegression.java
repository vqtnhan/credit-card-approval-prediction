package com.classifiers;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.UpperSPDDenseMatrix;
import no.uib.cipr.matrix.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class MyMultipleLinearRegression extends AbstractClassifier {
    protected double[] m_Coefficients;
    protected boolean[] m_SelectedAttributes;
    protected Instances m_TransformedData;
    protected ReplaceMissingValues m_MissingFilter;
    protected NominalToBinary m_TransformFilter;
    protected double m_ClassStdDev;
    protected double m_ClassMean;
    protected int m_ClassIndex;
    protected double[] m_Means;
    protected double[] m_StdDevs;
    protected int m_AttributeSelection;
    protected double m_Ridge = 1.0e-8;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        data = new Instances(data);

        m_TransformFilter = new NominalToBinary();
        m_TransformFilter.setInputFormat(data);
        data = Filter.useFilter(data, m_TransformFilter);
        m_MissingFilter = new ReplaceMissingValues();
        m_MissingFilter.setInputFormat(data);
        data = Filter.useFilter(data, m_MissingFilter);

        m_ClassIndex = data.classIndex();
        m_TransformedData = data;

        m_Coefficients = null;

        m_SelectedAttributes = new boolean[data.numAttributes()];
        m_Means = new double[data.numAttributes()];
        m_StdDevs = new double[data.numAttributes()];
        for (int j = 0; j < data.numAttributes(); ++j) {
            if (j != m_ClassIndex) {
                m_SelectedAttributes[j] = true;
                m_Means[j] = data.meanOrMode(j);
                m_StdDevs[j] = Math.sqrt(data.variance(j));
                if (m_StdDevs[j] == 0) {
                    m_SelectedAttributes[j] = false;
                }
            }
        }

        m_ClassStdDev = Math.sqrt(data.variance(m_TransformedData.classIndex()));
        m_ClassMean = data.meanOrMode(m_TransformedData.classIndex());

        findBestModel();

        m_TransformedData = new Instances(data, 0);
        return;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instance transformedInstance = instance;
        m_TransformFilter.input(transformedInstance);
        m_TransformFilter.batchFinished();
        transformedInstance = m_TransformFilter.output();
        m_MissingFilter.input(transformedInstance);
        m_MissingFilter.batchFinished();
        transformedInstance = m_MissingFilter.output();

        return regressionPrediction(transformedInstance, m_SelectedAttributes,
                m_Coefficients);
    }

    protected boolean deselectColinearAttributes(boolean[] selectedAttributes,
            double[] coefficients) {

        double maxSC = 1.5;
        int maxAttr = -1, coeff = 0;
        for (int i = 0; i < selectedAttributes.length; ++i) {
            if (selectedAttributes[i]) {
                double SC = Math.abs(coefficients[coeff] * m_StdDevs[i] / m_ClassStdDev);
                if (SC > maxSC) {
                    maxSC = SC;
                    maxAttr = i;
                }
                ++coeff;
            }
        }
        if (maxAttr >= 0) {
            selectedAttributes[maxAttr] = false;
            return true;
        }
        return false;
    }

    protected void findBestModel() throws Exception {
        do {
            m_Coefficients = doRegression(m_SelectedAttributes);
        } while (deselectColinearAttributes(m_SelectedAttributes, m_Coefficients));
        return;
    }

    protected double calculateSE(boolean[] selectedAttributes,
            double[] coefficients) throws Exception {

        double mse = 0;
        for (int i = 0; i < m_TransformedData.numInstances(); ++i) {
            double prediction = regressionPrediction(m_TransformedData.instance(i), selectedAttributes,
                    coefficients);
            double error = prediction - m_TransformedData.instance(i).classValue();
            mse += error * error;
        }
        return mse;
    }

    protected double regressionPrediction(Instance transformedInstance,
            boolean[] selectedAttributes, double[] coefficients) throws Exception {
        double result = 0;
        int column = 0;
        for (int j = 0; j < transformedInstance.numAttributes(); ++j) {
            if ((m_ClassIndex != j) && (selectedAttributes[j])) {
                result += coefficients[column] * transformedInstance.value(j);
                ++column;
            }
        }
        result += coefficients[column];

        return result;
    }

    protected double[] doRegression(boolean[] selectedAttributes)
            throws Exception {
        int numAttributes = 0;
        for (boolean selectedAttribute : selectedAttributes) {
            if (selectedAttribute) {
                ++numAttributes;
            }
        }

        Matrix independentTransposed = null;
        Vector dependent = null;
        if (numAttributes > 0) {
            independentTransposed = new DenseMatrix(numAttributes, m_TransformedData.numInstances());
            dependent = new DenseVector(m_TransformedData.numInstances());

            for (int i = 0; i < m_TransformedData.numInstances(); ++i) {
                Instance inst = m_TransformedData.instance(i);
                double sqrt_weight = Math.sqrt(inst.weight());
                int index = 0;
                for (int j = 0; j < m_TransformedData.numAttributes(); ++j) {
                    if (j == m_ClassIndex) {
                        dependent.set(i, inst.classValue() * sqrt_weight);
                    } else {
                        if (selectedAttributes[j]) {
                            double value = inst.value(j) - m_Means[j];

                            value /= m_StdDevs[j];

                            independentTransposed.set(index, i, value * sqrt_weight);

                            ++index;
                        }
                    }
                }
            }
        }

        double[] coefficients = new double[numAttributes + 1];
        if (numAttributes > 0) {
            Vector aTy = independentTransposed.mult(dependent, new DenseVector(numAttributes));
            Matrix aTa = new UpperSPDDenseMatrix(numAttributes).rank1(independentTransposed);
            independentTransposed = null;
            dependent = null;

            double ridge = m_Ridge;
            for (int i = 0; i < numAttributes; ++i) {
                aTa.add(i, i, ridge);
            }
            Vector coeffsWithoutIntercept = aTa.solve(aTy, new DenseVector(numAttributes));
            System.arraycopy(((DenseVector) coeffsWithoutIntercept).getData(), 0, coefficients, 0, numAttributes);

        }
        coefficients[numAttributes] = m_ClassMean;

        int column = 0;
        for (int i = 0; i < m_TransformedData.numAttributes(); ++i) {
            if ((i != m_TransformedData.classIndex()) && (selectedAttributes[i])) {
                coefficients[column] /= m_StdDevs[i];

                coefficients[coefficients.length - 1] -= coefficients[column] * m_Means[i];
                ++column;
            }
        }

        return coefficients;
    }
}
