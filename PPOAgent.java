package edu.boun.edgecloudsim.utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;
import java.util.Arrays;

public class PPOAgent implements RLAgent, Serializable {

    private static final long serialVersionUID = 1L;

    //-----------------------------
    // PPO Hyperparameters
    //-----------------------------

    private final int stateDim;
    private final int actionDim;

    private final double minAction;
    private final double maxAction;

    private final double gamma;
    private final double lambda;

    private final double clipEpsilon;

    private final int batchSize;

    private final Random rnd;

    //-----------------------------
    // Networks
    //-----------------------------

    private MultiLayerNetwork actor;

    private MultiLayerNetwork critic;

    //-----------------------------
    // Rollout Buffer
    //-----------------------------

    private RolloutBuffer buffer;
    private double actorLR = 1e-4;
    private double criticLR = 3e-4;

    private int ppoEpochs = 10;
    private static class RolloutBuffer {

        ArrayList<double[]> states = new ArrayList<>();

        ArrayList<Double> actions = new ArrayList<>();

        ArrayList<Double> rewards = new ArrayList<>();

        ArrayList<double[]> nextStates = new ArrayList<>();

        ArrayList<Boolean> dones = new ArrayList<>();

        ArrayList<Double> values = new ArrayList<>();

        ArrayList<Double> logProbs = new ArrayList<>();

        void clear() {

            states.clear();

            actions.clear();

            rewards.clear();

            nextStates.clear();

            dones.clear();

            values.clear();

            logProbs.clear();
        }

        int size() {
            return states.size();
        }
    }
    public PPOAgent(

            int stateDim,

            int actionDim,

            double minAction,

            double maxAction,

            int batchSize,

            double gamma,

            double lambda,

            double clipEpsilon,

            long seed)

    {

        this.stateDim = stateDim;

        this.actionDim = actionDim;

        this.minAction = minAction;

        this.maxAction = maxAction;

        this.batchSize = batchSize;

        this.gamma = gamma;

        this.lambda = lambda;

        this.clipEpsilon = clipEpsilon;

        this.rnd = new Random(seed);

        buffer = new RolloutBuffer();

        buildActor();

        buildCritic();
    }
    private void buildActor() {

        MultiLayerConfiguration conf =

            new NeuralNetConfiguration.Builder()

            .seed(123)

            .weightInit(WeightInit.XAVIER)

            .updater(new Adam(1e-4))

            .optimizationAlgo(
                    OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)

            .list()

            .layer(new DenseLayer.Builder()

                    .nIn(stateDim)

                    .nOut(128)

                    .activation(Activation.RELU)

                    .build())

            .layer(new DenseLayer.Builder()

                    .nOut(64)

                    .activation(Activation.RELU)

                    .build())

            .layer(new OutputLayer.Builder(

                    LossFunctions.LossFunction.MSE)

                    .activation(Activation.TANH)

                    .nOut(actionDim)

                    .build())

            .build();

        actor = new MultiLayerNetwork(conf);

        actor.init();
    }
    private void buildCritic() {

        MultiLayerConfiguration conf =

            new NeuralNetConfiguration.Builder()

            .seed(321)

            .weightInit(WeightInit.XAVIER)

            .updater(new Adam(3e-4))

            .list()

            .layer(new DenseLayer.Builder()

                    .nIn(stateDim)

                    .nOut(128)

                    .activation(Activation.RELU)

                    .build())

            .layer(new DenseLayer.Builder()

                    .nOut(64)

                    .activation(Activation.RELU)

                    .build())

            .layer(new OutputLayer.Builder(

                    LossFunctions.LossFunction.MSE)

                    .activation(Activation.IDENTITY)

                    .nOut(1)

                    .build())

            .build();

        critic = new MultiLayerNetwork(conf);

        critic.init();
    }
    private double gaussianNoise()
    {
        return rnd.nextGaussian() * 0.10;
    }
    @Override
    public double act(double[] state, boolean training)
    {
        INDArray s =
                Nd4j.create(state)
                .reshape(1,stateDim);

        INDArray output =
                actor.output(s,false);

        double action =
                output.getDouble(0);

        // tanh (-1,1)
        action =
                0.5*(action+1.0)
                *(maxAction-minAction)
                +minAction;

        // exploration
        if(training)
            action += gaussianNoise();

        // clip
        if(action<minAction)
            action=minAction;

        if(action>maxAction)
            action=maxAction;

        return action;
    }
    private double value(double[] state)
    {
        INDArray s =
                Nd4j.create(state)
                .reshape(1,stateDim);

        return critic.output(s,false)
                .getDouble(0);
    }
    private double logProbability(
            double action,
            double mean)
    {
        double sigma = 0.10;

        double diff =
                action-mean;

        return
                -0.5*
                diff*diff/
                (sigma*sigma);
    }
    @Override
    public void remember(

            double[] state,

            double action,

            double reward,

            double[] nextState,

            boolean done)

    {

        INDArray s =
                Nd4j.create(state)
                .reshape(1,stateDim);

        double mean =
                actor.output(s,false)
                .getDouble(0);

        buffer.states.add(state);

        buffer.actions.add(action);

        buffer.rewards.add(reward);

        buffer.nextStates.add(nextState);

        buffer.dones.add(done);

        buffer.values.add(
                value(state));

        buffer.logProbs.add(
                logProbability(
                        action,
                        mean));
    }
    private boolean shouldTrain()
    {
        return
                buffer.size()
                >=
                batchSize;
    }
    @Override
    public void train()
    {

        if(buffer.size()<batchSize)
            return;

        updatePolicy();

        buffer.clear();

    }
    private void updatePolicy()
    {
        double[] returns =
                computeReturns();

        double[] advantages =
                computeAdvantages();

        normalize(advantages);

        ppoUpdate(
                returns,
                advantages);
    }
    private double[] computeReturns()
    {
        int n = buffer.size();

        double[] returns = new double[n];

        double R = 0.0;

        for(int t=n-1;t>=0;t--)
        {
            if(buffer.dones.get(t))
                R = 0;

            R =
                buffer.rewards.get(t)
                +
                gamma*R;

            returns[t]=R;
        }

        return returns;
    }
    private double[] computeAdvantages()
    {
        int n = buffer.size();

        double[] adv = new double[n];

        double gae = 0.0;

        for(int t=n-1;t>=0;t--)
        {
            double value =
                    buffer.values.get(t);

            double nextValue;

            if(t==n-1)
                nextValue=0;
            else
                nextValue=
                    buffer.values.get(t+1);

            double delta =
                    buffer.rewards.get(t)
                    +
                    gamma*nextValue
                    -
                    value;

            gae =
                delta
                +
                gamma
                *
                lambda
                *
                gae;

            adv[t]=gae;
        }

        return adv;
    }
    private void normalize(double[] adv)
    {
        double mean=0;

        for(double x:adv)
            mean+=x;

        mean/=adv.length;

        double std=0;

        for(double x:adv)
            std+=(x-mean)*(x-mean);

        std=Math.sqrt(std/adv.length);

        for(int i=0;i<adv.length;i++)
            adv[i]=(adv[i]-mean)/(std+1e-8);
    }
    
    private void ppoUpdate(

            double[] returns,

            double[] advantages)
    {

        trainCritic(returns);

        trainActor(advantages);

    }
    private void trainCritic(
            double[] returns)
    {

        for(int i=0;i<buffer.size();i++)
        {
            INDArray input =
                    Nd4j.create(buffer.states.get(i))
                    .reshape(1,stateDim);

            INDArray label =
                    Nd4j.create(new double[]{
                            returns[i]
                    }).reshape(1,1);

            critic.fit(input,label);
        }

    }
    private void trainActor(
            double[] advantages)
    {

        for(int epoch=0;
            epoch<ppoEpochs;
            epoch++)
        {

            for(int i=0;
                i<buffer.size();
                i++)
            {

                INDArray state =
                        Nd4j.create(
                                buffer.states.get(i))
                        .reshape(1,stateDim);

                double oldAction =
                        buffer.actions.get(i);

                double oldLogProb =
                        buffer.logProbs.get(i);

                double adv =
                        advantages[i];

                double mean =
                        actor.output(
                                state,false)
                        .getDouble(0);

                double newLogProb =
                        logProbability(
                                oldAction,
                                mean);

                double ratio =
                        Math.exp(
                                newLogProb
                                -
                                oldLogProb);

                double clipped =
                        Math.max(
                                1-clipEpsilon,

                                Math.min(
                                        ratio,
                                        1+clipEpsilon));

                double target =
                        clipped*adv;

                INDArray label =
                        Nd4j.create(
                                new double[]{
                                        target
                                }).reshape(1,1);

                actor.fit(
                        state,
                        label);
            }

        }

    }
    @Override
    public void load(String dir) throws Exception {

        actor = MultiLayerNetwork.load(new File(dir, "actor.zip"), true);

        critic = MultiLayerNetwork.load(new File(dir, "critic.zip"), true);

    }
    @Override
    public void save(String dir) throws Exception {

        actor.save(
                new File(dir, "actor.zip"), true);

        critic.save(
                new File(dir, "critic.zip"), true);

    }
    public void updateTargets() {
    	
    }
}