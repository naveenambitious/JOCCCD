package edu.boun.edgecloudsim.utils;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.ops.transforms.Transforms;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;

public class SACAgent implements RLAgent{
	private final int stateDim;
	private final int actionDim;  // usually 1 for your problem
	private final double minAction, maxAction;
	private ComputationGraph actor;          // actor (mean, logstd outputs)
	private ComputationGraph critic1, critic2;
	private ComputationGraph targetCritic1, targetCritic2;
	private ReplayBuffer buffer;
	private int batchSize, warmupSteps, trainStepsPerCall;
	private double gamma, tau, alpha;
	private double learningrate = 0.01;
    public SACAgent(int stateDim,int actionDim,
                    double minAction, double maxAction,
                    int replayCapacity, int batchSize,
                    int warmupSteps, int trainStepsPerCall,
                    double gamma, double tau, double alpha, long seed) {
    	this.actionDim = actionDim;
        this.stateDim = stateDim;
        this.minAction = minAction;
        this.maxAction = maxAction;
        this.batchSize = batchSize;
        this.warmupSteps = warmupSteps;
        this.trainStepsPerCall = trainStepsPerCall;
        this.gamma = gamma;
        this.tau = tau;
        this.alpha = alpha;

        this.buffer = new ReplayBuffer(replayCapacity, stateDim);
        Nd4j.getRandom().setSeed(seed);

        buildNetworks(stateDim, actionDim, learningrate);
        hardUpdateTargets(); // target = online at start
    }

    private void buildNetworks(int stateDim, int actionDim, double learningRate) {
    int hiddenSize = 64;

    // ===== ACTOR (Gaussian policy) =====
    ComputationGraphConfiguration.GraphBuilder actorConf = new NeuralNetConfiguration.Builder()
            .seed(1234)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .graphBuilder()
            .addInputs("state")
            .addLayer("dense1", new DenseLayer.Builder().nIn(stateDim).nOut(hiddenSize)
                    .activation(Activation.RELU).build(), "state")
            .addLayer("dense2", new DenseLayer.Builder().nIn(hiddenSize).nOut(hiddenSize)
                    .activation(Activation.RELU).build(), "dense1")
            // Actor outputs: mean and log_std
            .addLayer("mean", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nIn(hiddenSize).nOut(actionDim)
                    .activation(Activation.IDENTITY).build(), "dense2")
            .addLayer("logstd", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nIn(hiddenSize).nOut(actionDim)
                    .activation(Activation.IDENTITY).build(), "dense2")
            .setOutputs("mean", "logstd");

    actor = new ComputationGraph(actorConf.build());
    actor.init();

    // ===== CRITIC Q1 =====
    ComputationGraphConfiguration.GraphBuilder q1Conf = new NeuralNetConfiguration.Builder()
            .seed(1234)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .graphBuilder()
            .addInputs("state", "action")
            .addVertex("concat", new MergeVertex(), "state", "action")
            .addLayer("dense1", new DenseLayer.Builder().nIn(stateDim + actionDim).nOut(hiddenSize)
                    .activation(Activation.RELU).build(), "concat")
            .addLayer("dense2", new DenseLayer.Builder().nIn(hiddenSize).nOut(hiddenSize)
                    .activation(Activation.RELU).build(), "dense1")
            .addLayer("qvalue", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nIn(hiddenSize).nOut(1)
                    .activation(Activation.IDENTITY).build(), "dense2")
            .setOutputs("qvalue");

    critic1 = new ComputationGraph(q1Conf.build());
    critic1.init();

    // ===== CRITIC Q2 =====
    ComputationGraphConfiguration.GraphBuilder q2Conf = new NeuralNetConfiguration.Builder()
            .seed(1234)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(learningRate))
            .graphBuilder()
            .addInputs("state", "action")
            .addVertex("concat", new MergeVertex(), "state", "action")
            .addLayer("dense1", new DenseLayer.Builder().nIn(stateDim + actionDim).nOut(hiddenSize)
                    .activation(Activation.RELU).build(), "concat")
            .addLayer("dense2", new DenseLayer.Builder().nIn(hiddenSize).nOut(hiddenSize)
                    .activation(Activation.RELU).build(), "dense1")
            .addLayer("qvalue", new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nIn(hiddenSize).nOut(1)
                    .activation(Activation.IDENTITY).build(), "dense2")
            .setOutputs("qvalue");

    critic2 = new ComputationGraph(q2Conf.build());
    critic2.init();

    // ===== TARGET CRITICS =====
    targetCritic1 = critic1.clone();
    targetCritic2 = critic2.clone();
    }

    private void hardUpdateTargets() {
    	targetCritic1.setParams(critic1.params().dup());
    	targetCritic2.setParams(critic2.params().dup());
    }

    @Override
    public double act(double[] state, boolean training) {
	        INDArray s = Nd4j.create(state).reshape(1, stateDim);

	        // Actor forward: returns μ and logσ
	        INDArray[] outputs = actor.output(s);
	        INDArray mu = outputs[0];
	        INDArray logStd = outputs[1];
	        INDArray std = Transforms.exp(logStd, true);

	        INDArray noise = Nd4j.randn(mu.shape()); // ε ~ N(0,1)
	        INDArray action = mu.add(std.mul(noise));
	        INDArray aTanh = Transforms.tanh(action, true);

	        // Rescale to [minAction, maxAction]
	        double a = aTanh.getDouble(0);
	        a = 0.5 * (a + 1.0) * (maxAction - minAction) + minAction;
	        return a;
    }

    @Override
	    public void remember(double[] s, double a, double r, double[] s2, boolean done) {
	        buffer.add(s, a, r, s2, done);
    }

    @Override
    public void train() {
        if (buffer.size() < Math.max(batchSize, warmupSteps)) return;

		    // numerical/clamping constants
		    final double LOG_STD_MIN = -20.0;
		    final double LOG_STD_MAX = 2.0;
		    final double EPS = 1e-6;
		    final int NUM_CANDIDATES = 10; // candidate actions per state for approximate actor update

		    for (int t = 0; t < trainStepsPerCall; t++) {
		        SampleBatch batch = buffer.sample(batchSize);

		        INDArray s  = Nd4j.create(batch.states).reshape(batchSize, stateDim);
		        INDArray a  = Nd4j.create(batch.actions).reshape(batchSize, actionDim);
		        INDArray r  = Nd4j.create(batch.rewards).reshape(batchSize, 1);
		        INDArray s2 = Nd4j.create(batch.nextStates).reshape(batchSize, stateDim);
		        INDArray d  = Nd4j.create(batch.dones).reshape(batchSize, 1);

		        // =========================
		        // 1) Compute targets for critics
		        // =========================

		        // actor(s2) -> mu2, logstd2
		        INDArray[] actorOutS2 = actor.output(false, s2);
		        INDArray mu2 = actorOutS2[0];           // shape: (batchSize, actionDim)
		        INDArray logStd2 = actorOutS2[1];       // shape: (batchSize, actionDim)

		        // clamp logStd for numerical stability
		        logStd2 = Transforms.max(Transforms.min(logStd2, LOG_STD_MAX), LOG_STD_MIN);
		        INDArray std2 = Transforms.exp(logStd2, true);

		        // sample epsilon and build pre-tanh action, then tanh-squash + rescale
		        INDArray eps2 = Nd4j.randn(mu2.shape());
		        INDArray preTanhA2 = mu2.add(std2.mul(eps2));              // raw action before tanh
		        INDArray tanhA2 = Transforms.tanh(preTanhA2, true);
		        INDArray a2 = tanhA2.mul(0.5).add(0.5).mul(maxAction - minAction).add(minAction);

		        // compute log pi(a2 | s2)
		        // gaussian log-prob (per-dim)
		        INDArray var2 = std2.mul(std2);
		        INDArray gaussTerm = preTanhA2.sub(mu2).mul(preTanhA2.sub(mu2)).div(var2).mul(-0.5)
		                .sub(logStd2).sub(0.5 * Math.log(2.0 * Math.PI)); // shape (batchSize,actionDim)
		        // sum over action dims
		        INDArray gaussLogProb = gaussTerm.sum(1).reshape(batchSize, 1); // (batchSize,1)

		        // tanh-squash correction: log(1 - tanh(x)^2 + eps)
		        INDArray correction = Transforms.log(Nd4j.ones(tanhA2.shape()).sub(Transforms.pow(tanhA2, 2)).add(EPS));
		        INDArray correctionSum = correction.sum(1).reshape(batchSize, 1);

		        INDArray logPiNext = gaussLogProb.sub(correctionSum); // (batchSize,1)

		        // target Q: use target critics
		        INDArray q1TargetNext = forwardCritic(targetCritic1, s2, a2); // (batchSize,1)
		        INDArray q2TargetNext = forwardCritic(targetCritic2, s2, a2); // (batchSize,1)
		        INDArray minQTargetNext = Transforms.min(q1TargetNext, q2TargetNext);

		        // y = r + gamma * ( minQ - alpha*logPiNext ) * (1 - done)
		        INDArray y = r.add(minQTargetNext.sub(logPiNext.mul(alpha)).mul(gamma).mul(d.rsub(1.0)));

		        // =========================
		        // 2) Update critics (MSE)
		        // =========================
		        critic1.fit(new INDArray[]{s, a}, new INDArray[]{y});
		        critic2.fit(new INDArray[]{s, a}, new INDArray[]{y});

		        // =========================
		        // 3) Approximate actor update (candidate-sampling + supervised fit)
		        //    - For each sample in the batch, sample several candidate actions from π(s)
		        //    - Evaluate score = min(Q1,Q2) - alpha*logπ and pick best candidate
		        //    - Supervise actor outputs (mean, logstd) toward the chosen pre-tanh raw value
		        // Note: this is an approximation and avoids cross-graph backprop.
		        // =========================

		        INDArray[] actorOutS = actor.output(false, s);
		        INDArray mu = actorOutS[0].dup();       // (batchSize, actionDim)
		        INDArray logStd = actorOutS[1].dup();
		        logStd = Transforms.max(Transforms.min(logStd, LOG_STD_MAX), LOG_STD_MIN);
		        INDArray std = Transforms.exp(logStd, true);

		        INDArray targetMeans = Nd4j.create(mu.shape());       // will hold chosen pre-tanh targets
		        INDArray targetLogStd = logStd.dup();                 // keep logStd as-is (or you can shrink)

		        // loop over each sample in batch (actionDim typically small, usually 1)
		        for (int i = 0; i < batchSize; i++) {
		            INDArray muRow = mu.getRow(i).reshape(1, actionDim);            // (1,actionDim)
		            INDArray logStdRow = logStd.getRow(i).reshape(1, actionDim);
		            INDArray stdRow = std.getRow(i).reshape(1, actionDim);
		            INDArray sRow = s.getRow(i).reshape(1, stateDim);

		            double bestScore = -Double.MAX_VALUE;
		            INDArray bestPreTanh = null;

		            for (int k = 0; k < NUM_CANDIDATES; k++) {
		                INDArray epsRow = Nd4j.randn(1, actionDim);
		                INDArray pre = muRow.add(stdRow.mul(epsRow));               // pre-tanh candidate
		                INDArray tanhPre = Transforms.tanh(pre, true);
		                INDArray aCand = tanhPre.mul(0.5).add(0.5).mul(maxAction - minAction).add(minAction);

		                // gaussian log-prob (sum dims)
		                INDArray varRow = stdRow.mul(stdRow);
		                INDArray gaussTermRow = pre.sub(muRow).mul(pre.sub(muRow)).div(varRow).mul(-0.5)
		                        .sub(logStdRow).sub(0.5 * Math.log(2.0 * Math.PI));
		                double gaussLog = gaussTermRow.sumNumber().doubleValue();

		                // correction term
		                INDArray corrRow = Transforms.log(Nd4j.ones(tanhPre.shape()).sub(Transforms.pow(tanhPre, 2)).add(EPS));
		                double corrSum = corrRow.sumNumber().doubleValue();

		                double logPi = gaussLog - corrSum;

		                // evaluate candidate via target critics (use target critics for stability)
		                INDArray q1Row = forwardCritic(targetCritic1, sRow, aCand);
		                INDArray q2Row = forwardCritic(targetCritic2, sRow, aCand);
		                double minQ = Math.min(q1Row.getDouble(0), q2Row.getDouble(0));

		                double score = minQ - alpha * logPi; // objective we want to maximize
		                if (score > bestScore) {
		                    bestScore = score;
		                    bestPreTanh = pre.dup(); // store the chosen pre-tanh (raw) value
		                }
		            } // end candidates loop

		            // write chosen pre-tanh to targets
		            targetMeans.putRow(i, bestPreTanh);
		            // targetLogStd keep as current logStdRow (or adjust if you want)
		        } // end batch loop

		        // supervise actor to move its outputs to (targetMeans, targetLogStd)
		        // actor expects two outputs: mean and logstd
		        actor.fit(new INDArray[]{s}, new INDArray[]{targetMeans, targetLogStd});

		        // =========================
		        // 4) Soft-update target critics
		        // =========================
		        softUpdate(targetCritic1, critic1, tau);
		        softUpdate(targetCritic2, critic2, tau);
    } // end trainStepsPerCall loop
    }

    private static void softUpdate(ComputationGraph target, ComputationGraph online, double tau) {
	        INDArray tp = target.params();
	        INDArray op = online.params();
	        INDArray newParams = tp.mul(1.0 - tau).add(op.mul(tau));
	        target.setParams(newParams);
    }

    private static INDArray forwardCritic(ComputationGraph net, INDArray s, INDArray a) {
        // DL4J multi-input forward:
        INDArray[] outs = net.output(false, s, a);
        return outs[0];
    }
    @Override
    public void load(String path) {
        // TODO: load model parameters from file if needed
        System.out.println("Load method not yet implemented: " + path);
    }
    @Override
    public void updateTargets() { 
    	// TODO: load model parameters from file if needed
        System.out.println("updateTargets method not yet implemented: ");
    }

    @Override
    public void save(String dir) throws Exception {
    	// TODO: load model parameters from file if needed
        System.out.println("save method not yet implemented: ");
    }
}
