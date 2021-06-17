import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Xor {

    final static int fit = 1000;

    public static void main(String[] args) {
        //1. configuration
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
            .miniBatch(false)
            .updater(new Sgd(0.1))
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(2)
                .nOut(4)
                .weightInit(new UniformDistribution(0, 1))
                .activation(Activation.SIGMOID)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(4)
                .nOut(2).activation(Activation.SOFTMAX)
                .weightInit(new UniformDistribution(0, 1)).build())
            .build();
        //2. network
        MultiLayerNetwork network = new MultiLayerNetwork(configuration);
        /**
         * 1,0 - true (0, 1)
         * 0, 1 - false (0, 1)
         * 0,0 - false (1 , 0)
         * 1, 1 - false (1, 0)
         */
        //3. prepare dataset
        double[][] features = new double[][] {
            {0, 0},
            {1, 0},
            {0, 1},
            {1, 1}
        };

        double[][] labels = new double[][] {
            {1, 0},
            {0, 1},
            {0, 1},
            {1, 0}
        };
        DataSet ds = new DataSet(Nd4j.create(features), Nd4j.create(labels));

        //4. fit - 1000 epochs

        for (int i = 0; i < fit; i++) {
            network.fit(ds);
        }

        //5. evaluation/prediction
        Evaluation evaluation = new Evaluation();
        evaluation.eval(ds.getLabels(), network.output(ds.getFeatures()));
        System.out.println(evaluation.stats());

        // predict
        System.out.println(network.output(Nd4j.create(new double[][] {
            {1, 0}
        })));
    }
}
