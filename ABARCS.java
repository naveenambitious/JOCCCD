package edu.boun.edgecloudsim.utils;




public class ABARCS {

	private int[][][] Ql;
	private int[][][][] deadlineStats;
	private static ABARCS instance = null;
	public static ABARCS getInstance() {
			if(instance == null) {
				instance = new ABARCS();
			}
			return instance;
	}

	public void ABARCSInitialize(int numOfAccessPoint, int total_slots, int taskTypes) {
	    	Ql = new int[numOfAccessPoint][total_slots][taskTypes];
	    	for(int apIndex=0; apIndex<numOfAccessPoint; apIndex++) {
				for(int slot=0;slot<total_slots; slot++) {
					for(int tasktype=0;tasktype<taskTypes; tasktype++) {
								Ql[apIndex][slot][tasktype]=0;
				}
			}
	}
	    	deadlineStats = new int[numOfAccessPoint][total_slots][taskTypes][2];
	    	for(int ap=0; ap<numOfAccessPoint; ap++)
	    	{
	    	    for(int slot=0; slot<total_slots; slot++)
	    	    {
	    	        for(int type=0; type<taskTypes; type++)
	    	        {
	    	            deadlineStats[ap][slot][type][0] = 0; // Miss Count
	    	            deadlineStats[ap][slot][type][1] = 0; // Completed Count
	    	        }
	    	    }
	    	}
}


	public int[][][] getQl(){
		return Ql;
	}
	public int [][][][]getht(){
		return deadlineStats;
	}
	public double getDeadlineMissRatio(
	        int apId,
	        int slot)
	{
	    int miss = 0;
	    int completed = 0;

	    for(int s=0; s<=slot; s++)
	    {
	        for(int type=0;
	            type<deadlineStats[apId][s].length;
	            type++)
	        {
	            miss +=
	                deadlineStats[apId][s][type][0];

	            completed +=
	                deadlineStats[apId][s][type][1];
	        }
	    }

	    if(completed==0)
	        return 0.0;

	    return
	        (double)miss
	        /
	        completed;
	}
	public void updateHT(
	        int apId,
	        int slot,
	        int taskType,
	        boolean deadlineMiss)
	{
	    // completed task
		deadlineStats[apId][slot][taskType][1]++;

	    // missed deadline
	    if(deadlineMiss)
	    {
	        deadlineStats[apId][slot][taskType][0]++;
	    }
	}
	public void updateQl(int apId, int slot, int taskType){
			if (Ql == null) {
			        throw new IllegalStateException("Ql not initialized!");
    			}
    			Ql[apId][slot][taskType]++;
    }
	public int getTotalTasks() {
	    int total = 0;
	    if (Ql != null) {
	        for (int i = 0; i < Ql.length; i++) {               // iterate over apIndex
	            for (int j = 0; j < Ql[i].length; j++) {        // iterate over slot
	                for (int k = 0; k < Ql[i][j].length; k++) { // iterate over taskType
	                    total += Ql[i][j][k];
	                }
	            }
	        }
	    }
	    return total;
	}
	public void printAllTasks() {
	    if (Ql == null) {
	        System.out.println("Ql is not initialized!");
	        return;
	    }

	    for (int ap = 0; ap < Ql.length; ap++) {
	        System.out.println("=== AP ID: " + ap + " ===");
	        for (int slot = 0; slot < Ql[ap].length; slot++) {
	            System.out.println("  Slot: " + slot);
	            for (int taskType = 0; taskType < Ql[ap][slot].length; taskType++) {
	                System.out.print("    TaskType " + taskType + " -> " + Ql[ap][slot][taskType]);
	            }
	        }
	    }
	}
	public double getArrivalRate(
	        int apId,
	        int slot,
	        int taskType,
	        double slotDuration) {
		return Ql[apId][slot][taskType]/slotDuration;
	
	
	}
}
