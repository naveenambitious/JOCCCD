package edu.boun.edgecloudsim.utils;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;


public class RewardLogger {

	private PrintWriter writer;

	    public RewardLogger(String filename) {
	        try {
	            writer = new PrintWriter(new FileWriter(filename, true));
	            writer.println("epoch,slot,time,apId,reward,action,delay,throughput,deadlineMiss,q1,q2,lambda1,lambda2,bw1,bw2"); // header
	        } catch (IOException e) {
	        	System.out.println("error reading file...");
	            e.printStackTrace();
	        }
	    }

	    public void log1(int epoch, double reward, int apId) {
	    	System.out.println(epoch + "," + reward + "," + apId);
	        //writer.println(epoch + "," + reward + "," + apId);
	    }
	    public void log(int epoch,int slot,double time,int apId,double reward,double action,double delay,double throughput,double deadlineMiss,double q1,double q2,double lambda1,double lambda2,double bw1,double bw2) {
	      writer.println(
	            	        epoch + "," +
	            	        slot + "," +
	            	        time + "," +
	            	        apId + "," +
	            	        reward + "," +
	            	        action + "," +
	            	        delay + "," +
	            	        throughput + "," +
	            	        deadlineMiss + "," +
	            	        q1 + "," +
	            	        q2 + "," +
	            	        lambda1 + "," +
	            	        lambda2 + "," +
	            	        bw1 + "," +
	            	        bw2
	            	    );
	    	
	    }
	    public void close() {
	        writer.close();
    }
}
