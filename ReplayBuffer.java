package edu.boun.edgecloudsim.utils;

import java.util.Random;


public class ReplayBuffer {
	private final int capacity;
    private final int stateDim;

    private final double[][] states;
    private final double[] actions;
    private final double[] rewards;
    private final double[][] nextStates;
    private final boolean[] dones;

    private int size = 0;
    private int idx = 0;
    private final Random rnd = new Random(123);

    public ReplayBuffer(int capacity, int stateDim) {
        this.capacity = capacity;
        this.stateDim = stateDim;
        states = new double[capacity][stateDim];
        actions = new double[capacity];
        rewards = new double[capacity];
        nextStates = new double[capacity][stateDim];
        dones = new boolean[capacity];
    }

    public void add(double[] s, double a, double r, double[] s2, boolean done) {
        System.arraycopy(s, 0, states[idx], 0, stateDim);
        actions[idx] = a;
        rewards[idx] = r;
        System.arraycopy(s2, 0, nextStates[idx], 0, stateDim);
        dones[idx] = done;

        idx = (idx + 1) % capacity;
        size = Math.min(size + 1, capacity);
    }

    public SampleBatch sample(int batch) {
        int n = Math.min(batch, size);
        double[] actionsB = new double[n];
        double[] rewardsB = new double[n];
        double[] donesB = new double[n];
        double[][] statesB = new double[n][stateDim];
        double[][] nextStatesB = new double[n][stateDim];

        for (int i = 0; i < n; i++) {
            int j = rnd.nextInt(size);
            System.arraycopy(states[j], 0, statesB[i], 0, stateDim);
            actionsB[i] = actions[j];
            rewardsB[i] = rewards[j];
            System.arraycopy(nextStates[j], 0, nextStatesB[i], 0, stateDim);
            donesB[i] = dones[j] ? 1.0 : 0.0;
        }
        return new SampleBatch(statesB, actionsB, rewardsB, nextStatesB, donesB);
    }

    public int size() { return size; }
}
class SampleBatch {
    public final double[][] states, nextStates;
    public final double[] actions, rewards, dones;
    public SampleBatch(double[][] s, double[] a, double[] r, double[][] s2, double[] d) {
        this.states = s; this.actions = a; this.rewards = r; this.nextStates = s2; this.dones = d;
    }
}
