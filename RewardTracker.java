package edu.boun.edgecloudsim.utils;

import java.util.*;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.File;


public class RewardTracker {
    // Map: apId -> list of rewards over time
    private Map<Integer, PerformanceMetrics> apMetrics;
    private static RewardTracker instance = null;
    private File file;
    private PrintWriter writer;
    public static RewardTracker getInstance() {
		if(instance == null) {
			instance = new RewardTracker();
		}
		return instance;
	}
    public void RewardTrackerInitialize(int numOfAccessPoints) {
    	apMetrics = new HashMap<>();
        for (int i = 0; i < numOfAccessPoints; i++) {
        	apMetrics.put(i, new PerformanceMetrics());
        }
        try {

            file = new File("APSummary.csv");
            boolean exists = file.exists();

            writer = new PrintWriter(
                    new FileWriter(file, true));

            if(!exists){
                writer.println(
                "iteration,num_devices,approach,apId,avg_reward,avg_delay,avg_throughput,Varience,JainFairness");
            }

        } catch(IOException e){
            e.printStackTrace();
        }
	}
        // Store reward for a given AP
    public void recordMetrics(int apId, double reward, double delay, double throughput) {
        PerformanceMetrics metrics = apMetrics.get(apId);
        metrics.addReward(reward);
        metrics.addDelay(delay);
        metrics.addThroughput(throughput);
    }

		    // Print summary
    public void printSummary(
            int iteration,
            int numDevices,
            String approach)
    {

        for(Map.Entry<Integer, PerformanceMetrics> entry
                : apMetrics.entrySet()) {

            int apId = entry.getKey();
            PerformanceMetrics m = entry.getValue();

            writer.println(
                    iteration + "," +
                    numDevices + "," +
                    approach + "," +
                    apId + "," +
                    m.getAverageReward() + "," +
                    m.getAverageDelay() + "," +
                    m.getAverageThroughput()+ "," +
                    calculateVariance()+ "," +
                    calculateJainsFairness());
        }

        writer.flush();   // ensure data is written
    }
		    public double calculateVariance() {
		    	List<Double> rewards = apMetrics.values().stream()
		                .map(PerformanceMetrics::getAverageReward)
		                .filter(r -> r != null && !r.isNaN() && !r.isInfinite())
		                .collect(Collectors.toList());
		        double mean = rewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
		        double variance = rewards.stream()
		                                 .mapToDouble(r -> Math.pow(r - mean, 2))
		                                 .average()
		                                 .orElse(0.0);
		        return variance;
		    }
		    public double calculateJainsFairness() {
		    	List<Double> rewards = apMetrics.values().stream()
		                .map(PerformanceMetrics::getAverageReward)
		                .filter(r -> r != null && !r.isNaN() && !r.isInfinite())
		                .collect(Collectors.toList());
		    	if (rewards.isEmpty()) {
		            return 0.0; // no valid rewards
		        }
		        double sum = rewards.stream().mapToDouble(Double::doubleValue).sum();
		        double sumSq = rewards.stream().mapToDouble(r -> r * r).sum();
		        int n = rewards.size();
		        
		        if (sumSq == 0) return 0.0; // avoid div by zero
		        return (sum * sum) / (n * sumSq);
		    }
		}

class PerformanceMetrics {
    private List<Double> rewards;
    private List<Double> delays;
    private List<Double> throughputs;

    public PerformanceMetrics() {
        rewards = new ArrayList<>();
        delays = new ArrayList<>();
        throughputs = new ArrayList<>();
    }

    public void addReward(double reward) {
        rewards.add(reward);
    }

    public void addDelay(double delay) {
        delays.add(delay);
    }

    public void addThroughput(double throughput) {
        throughputs.add(throughput);
    }

    public double getAverageReward() {
        return rewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    public double getAverageDelay() {
        return delays.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    public double getAverageThroughput() {
        return throughputs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    public int getNumSamples() {
        return rewards.size(); // assuming all lists grow equally
    }
    
}