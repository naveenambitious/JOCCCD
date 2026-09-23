package edu.boun.edgecloudsim.utils;

import java.util.Random;

public class OUNoise {
	private final double mu, theta, sigma;
    private double xPrev = 0.0;
    private final Random rnd;

    public OUNoise(double mu, double theta, double sigma, long seed) {
        this.mu = mu; this.theta = theta; this.sigma = sigma;
        this.rnd = new Random(seed);
    }

    public double next() {
        double dx = theta * (mu - xPrev) + sigma * rnd.nextGaussian();
        xPrev += dx;
        return xPrev;
    }

    public void reset() { xPrev = 0.0; }
}
