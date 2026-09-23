package edu.boun.edgecloudsim.utils;

public interface RLAgent {
	// Choose action for current state (no gradient update)
    double act(double[] state, boolean training);

    // Store transition
    void remember(double[] state, double action, double reward, double[] nextState, boolean done);

    // Perform one or more gradient steps
    void train();

    // (Optional) hard/soft update of target networks on demand
    void updateTargets();

    // Save/Load
    void save(String dir) throws Exception;
    void load(String dir) throws Exception;
}
