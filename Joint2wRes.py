import simpy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import sys

cacheHits= 0
cacheMiss = 0
total_data_migration_time =list()
#Creating a blank Dataframe with attributes 'Dev_id', 'Job_id', 'Arraival_time', 'Execution_time'
Record = pd.DataFrame(columns=['Dev_id', 'Job_id', 'Arraival_time', 'Execution_time', 'Status'])
Record1 = pd.DataFrame(columns=['Dev_id', 'Job_id', 'Execution_time','waiting_time','task_status'])
class BaseStation(object):
    def __init__(self, env, name, location, rcd):
        self.env = env
        self.name = name
        self.location = location
        self.queue_tasks = simpy.Store(env)
        self.cores = 20  # Number of CPU cores
        self.instruction_per_cyle =50
        self.clock_speed = 1.2 * 10**9
        self.clock_cycle_time = 1/ (1.2 * (10**9))
        self.active_tasks = 0  # Number of currently active tasks
        self.RCD = rcd
    def remove_task(self, dataCachingRecord):
        while True:
            if self.active_tasks < self.cores:
                self.active_tasks += 1
                task = yield self.queue_tasks.get()
                self.env.process(self.execute_task(task))
            else:
                yield self.env.timeout(0.001)  # Wait a short time before checking again

    def execute_task(self, task):
        ##-print(f"Base Station Id : {self.name.split('_')[1]}")
        WT= self.env.now
        #print(f"Base station name {self.name} and Task name {task.name_prefix}")
        cache_location_BS = dataCacheLocation[int(task.name_prefix.split('_')[1])]
        #print(f"Identified Cache location is {cache_location_BS.name.split('_')[1]}")
        if int(self.name.split('_')[1]) == int(cache_location_BS.name.split('_')[1]):
            global cacheHits
            cacheHits = cacheHits+1
            yield self.env.timeout( (task.task_size/self.instruction_per_cyle) * (self.clock_cycle_time))  # Execution time for same base station
            currentTime = self.env.now
            global Record
            # Convert Dev_id and Job_id to integers
            Record['Dev_id'] = Record['Dev_id'].astype(int)
            Record['Job_id'] = Record['Job_id'].astype(int)
            #print(f"======== {self.env.now}", end=" ")
            dev_id_criteria = int(task.name_prefix.split('_')[1])
            job_id_criteria = int(task.task_counter)
            ##-print(f"Testing Dev Id and Task Id {dev_id_criteria} and {job_id_criteria}")
            # Update the execution_time based on the criteria
            ##-print(f"Time : {self.env.now}")
            #Record.loc[(Record['Dev_id'] == dev_id_criteria) & (Record['Job_id'] == job_id_criteria), 'Execution_time'] = currentTime
            self.RCD.loc[len(self.RCD)] = {'Dev_id': int(task.name_prefix.split('_')[1]), 'Job_id': int(task.task_counter), 'Execution_time': currentTime, 'waiting_time': WT,'task_status': 'Completed'}
            #print(f"task comleted at base station {task.name_prefix}_{task.task_counter}")
            #self.RCD.loc[self.RCD['Dev_id'] == int(task.name_prefix.split('_')[1]) & self.RCD['Job_id']== int(task.task_counter), 'Execution_time'] = currentTime
            #print(Record.loc[(Record['Dev_id'] == dev_id_criteria) & (Record['Job_id'] == job_id_criteria), 'Arraival_time'])# == self.env.now
            ##-print(f"Task {task.name_prefix}_{task.task_counter} started at {task.start_time} ended at {task.end_time}")
            task.end_time = self.env.now
            task.total_execution_time = task.end_time - task.start_time
            task.results.append(task.total_execution_time)
        else:
            #remove from current base station and load it to attached base station
            # task migration time
            Data_migration_time = ((self.distance_BSs(cache_location_BS.location, self.location))/ (2.4 * 10**8)) + ( task.task_size/ (10**6)) 
            #print(f"Data Migiration time is {Data_migration_time} and distance is {self.distance_BSs(cache_location_BS.location, self.location)}")
            total_data_migration_time.append(Data_migration_time)
            yield self.env.timeout(Data_migration_time)
            cache_location_BS.queue_tasks.put(task)
            #print(f"task name migrated to another base station {task.name_prefix}_{task.task_counter}")
            #Data_transition_time = distance_BSs(cache_location_BS.location, self.location)
            global cacheMiss
            cacheMiss = cacheMiss +1
            #yield self.env.timeout(( (task.task_size/self.instruction_per_cyle) * (self.clock_cycle_time)) + Data_transition_time)  # Execution + data transition time
        self.active_tasks -= 1  # Mark task as completed
        
        
        
    '''
    def remove_tasks(self, dataCachingRecord):
        while True:
            if len(self.queue_tasks.items) >= self.num_cores:
                tasks = [self.queue_tasks.get() for _ in range(self.num_cores)]
                tasks = yield self.env.all_of(tasks)
                yield self.env.process(self.execute_tasks(list(tasks.values())))
            else:
                yield self.env.timeout(0.01)  # Check again after 1 time unit

    def execute_tasks(self, tasks):
        for task in tasks:
            print(f"Base Station Id : {self.name.split('_')[1]}")
            if int(self.name.split('_')[1]) == int(task.name.split('_')[1]):
                yield self.env.timeout( (task.task_size/self.instruction_per_cyle) * (self.clock_cycle_time))
                
            else:
                Data_transition_time = 0.02
                yield self.env.timeout(( (task.task_size/self.instruction_per_cyle) * (self.clock_cycle_time)) + Data_transition_time)
            task.end_time = self.env.now
            task.total_execution_time = task.end_time - task.start_time
            task.results.append(task.total_execution_time)
    '''
    '''  
    def remove_task(self, dataCachingRecord):
        task = yield self.queue_tasks.get()
        print(f"Base Station Id : {self.name.split('_')[1]}")
        if int(self.name.split('_')[1]) == int(task.name.split('_')[1]):
            yield self.env.timeout(0.006)
        else:
            Data_transition_time = 0.02
            yield self.env.timeout(0.006 + Data_transition_time)
        task.end_time = self.env.now
        task.total_execution_time = task.end_time - task.start_time
        task.results.append(task.total_execution_time)
        
    '''
    def execute_task1(self, task):
        print("Executed")
    def distance_BSs(self,loc_current, loc_cacheData):
        return euclidean_distance(loc_current , loc_cacheData)

class RelayAgent(object):
    def __init__(self, env, name, location):
        self.env = env
        self.name = name
        self.location = location
        self.base_station = None
        self.queue_type_1 = simpy.Store(env)
        self.queue_type_2 = simpy.Store(env)
        self.bandwidth = 100  # Mbps
        self.lastSlot = 0.0
        self.Q_t_F = 60
        self.Q_t_S = 40

    def getBandwidth(self, type_task):
        if self.env.now - self.lastSlot > 2.0:
            Q_t_I = len(self.queue_type_1.items) * 0.6 / 1100
            Q_t_II = len(self.queue_type_2.items) * 0.4 / 1100
            try:
                if (Q_t_I == 0 or Q_t_II ==0):
                           if type_task == 1:
                                      return self.Q_t_F
                           else:
                                      return self.Q_t_S
                else:
                           
                           if Q_t_II < Q_t_I:
                               self.lastSlot = self.env.now
                               if type_task == 1:
                                   self.Q_t_F = self.bandwidth * Q_t_I / (Q_t_I + Q_t_II)
                                   self.Q_t_S = self.bandwidth * 0.8 * Q_t_II / (Q_t_I + Q_t_II)
                                   return self.Q_t_F
                               else:
                                   self.Q_t_S = self.bandwidth * Q_t_I / (Q_t_I + Q_t_II)
                                   self.Q_t_F = self.bandwidth * 0.8 * Q_t_II / (Q_t_I + Q_t_II)
                                   return self.Q_t_S
                           else:
                               self.lastSlot = self.env.now
                               if type_task == 1:
                                   self.Q_t_S = self.bandwidth * Q_t_I / (Q_t_I + Q_t_II)
                                   self.Q_t_F = self.bandwidth * 0.8 * Q_t_II / (Q_t_I + Q_t_II)
                                   return self.Q_t_F
                               else:
                                   self.Q_t_F = self.bandwidth * Q_t_I / (Q_t_I + Q_t_II)
                                   self.Q_t_S = self.bandwidth * 0.8 * Q_t_II / (Q_t_I + Q_t_II)
                                   return self.Q_t_S
            except ValueError:
                print("Invalid value")
        else:
            if type_task == 1:
                return self.Q_t_F
            else:
                return self.Q_t_S

    def remove_task(self, queue_type, BSstations):
        if queue_type == 1:
            task = yield self.queue_type_1.get()
            ##-print(f"Removing task from queue_type_1 of {self.name}")
        elif queue_type == 2:
            task = yield self.queue_type_2.get()
            ##-print(f"Removing task from queue_type_2 of {self.name}")
        else:
            raise ValueError("Invalid queue type")
        attached_base_station_number = min(BSstations, key=lambda BSstations: euclidean_distance(self.location, BSstations.location))
        ##-print(f"Attached base station is {attached_base_station_number.name}")
        trans_time = (1024 * 1000) / ((self.getBandwidth(queue_type) * math.log2(1 + 45.86)) * 1000)
        yield self.env.timeout(trans_time)
        task.start_time = self.env.now
        attached_base_station_number.queue_tasks.put(task)

dataCacheLocation = dict()
RelayAttachedBaseStation = dict()

class DataCaching(object):
    def __init__(self, env, dev_no, baseStation):
        self.env = env
        if dev_no not in dataCacheLocation.keys():
            dataCacheLocation[dev_no] = baseStation
        else:
            dataCacheLocation[dev_no] = baseStation

    @staticmethod
    def getDICT():
        return dataCacheLocation

def remove_tasks_from_agents(env, agents, baseStationS):
    for agent in agents:
        env.process(agent.remove_task(1, baseStationS))
        env.process(agent.remove_task(2, baseStationS))

def remove_tasks_from_BaseStations(env, baseStationS, dataCachingRecord):
    for bs in baseStationS:
        env.process(bs.remove_task(dataCachingRecord))

def trigger_task_removal(env, agents, baseStationS):
    yield env.timeout(1)
    remove_tasks_from_agents(env, agents, baseStationS)

def trigger_task_execute(env, baseStationS, dataCachingRecord):
    yield env.timeout(1.5)
    remove_tasks_from_BaseStations(env, baseStationS, dataCachingRecord)

def traverseBaseStation():
    pass


class GenTask(object):
    results = []

    def __init__(self, env, dev_id, name_prefix, prio, location, ra, bs,rcd = Record):
        self.env = env
        self.dev_id = dev_id
        self.name_prefix = name_prefix
        self.prio = prio
        self.location = location
        self.relay_agents = ra
        self.base_stations = bs
        self.task_counter = 0
        self.action = env.process(self.generator_task())
        self.closetRA = None
        self.start_time = None
        self.task_size = random.randint(1024, 8192)
        self.record= Record
        self.speed_x = 2
        self.speed_y =2
        self.active_time =env.now
    def generator_task(self):
        while True:
                   self.task_counter += 1
                   task_name = f"{self.name_prefix}_{self.task_counter}"
                   arr = self.arrival_time()
                   #global Record
                   self.record.loc[len(self.record)] = {'Dev_id': int(self.name_prefix.split('_')[1]), 'Job_id': int(self.task_counter), 'Arraival_time': self.env.now, 'Execution_time': 0.0, 'Status':'Not Completed'}
                   yield self.env.timeout(arr)
                   
                   self.start_time = arr
                   ##-print("Task " + task_name + " arrived at " + str(self.env.now))
                   ##-print("Task " + task_name + " location is " + str(self.location[0]) + " and " + str(self.location[1]))
                   self.move()
                   closest_relay_agent = min(self.relay_agents, key=lambda ra: euclidean_distance(self.location, ra.location))
                   self.closetRA = closest_relay_agent
                   ##-print("Task " + task_name + " Relay agent is " + closest_relay_agent.name)
                   
                   if self.prio == 1:
                       yield self.env.timeout(0.01)
                       self.closetRA.queue_type_1.put(self)
                   else:
                       self.closetRA.queue_type_2.put(self)
                       yield self.env.timeout(0.01)
                   
                   ##-print(f"{task_name} is loaded to {self.closetRA.name}")
                   self.closetRA.base_station.queue_tasks.put(self)
    def move(self):
        new_x = self.location[0] + self.speed_x * (self.active_time - self.env.now)
        new_y = self.location[1] + self.speed_y * (self.active_time - self.env.now)

        if new_x <= 0 or new_x >= 1000:
            self.speed_x = -self.speed_x
        

        if new_y <= 0 or new_y >= 1000:
            self.speed_y = -self.speed_y
        if ((new_x <= 0 or new_x >= 1000) or (new_y <= 0 or new_y >= 1000) ):
            pass
        else:
            self.location= (new_x, new_y)

    def arrival_time(self):
        if self.prio == 1:
            return np.random.uniform(5.7)
        elif self.prio == 2:
            return np.random.uniform(8.8)
        else:
            print("Wrong task type")
            return -1

def euclidean_distance(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)
def plot_entities(base_stations, relay_agents, tasks,stat):
    count=0
    plt.figure(figsize=(10, 10))
    # Plot Base Stations
    for bs in base_stations:
        plt.scatter(bs.location[0], bs.location[1], c='red', marker='s', label='Base Station' if bs == base_stations[0] else "")
    # Plot Relay Agents
    for ra in relay_agents:
        plt.scatter(ra.location[0], ra.location[1], c='blue', marker='o', label='Relay Agent' if ra == relay_agents[0] else "")
    # Plot Tasks
    for task in tasks:
        count = count+1
        plt.scatter(task.location[0], task.location[1], c='green', marker='x', label='Device' if task == tasks[0] else "")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.title("Simulation Entities on 1000x1000 Grid")
    plt.grid(True)
    #plt.show()
    plt.savefig(f'{count}_{stat}.png')
def main():
    env = simpy.Environment()
    
    bandwidth = 100  # Mbps
    tasktype = [1, 2]
    pvalues = [0.25, 0.75]
    BS_count = 20
    Relay_count = 40
    dev_num = int(sys.argv[1])
    base_stations = [BaseStation(env, f"BaseStation_{i}", (np.random.randint(1000), np.random.randint(1000)), Record1) for i in range(BS_count)]
    relay_agents = [RelayAgent(env, f"RelayAgent_{i}", (np.random.randint(1000), np.random.randint(1000))) for i in range(Relay_count)]
    devices =[]
    for relay_agent in relay_agents:
        closest_base_station = min(base_stations, key=lambda bs: euclidean_distance(relay_agent.location, bs.location))
        relay_agent.base_station = closest_base_station
    '''
    for relay_agent in relay_agents:
        print(f"{relay_agent.name} is closest to {relay_agent.base_station.name}")
    '''
    for x in range(dev_num):
        pr = np.random.choice(tasktype, p=pvalues)
        loc = (np.random.randint(1000), np.random.randint(1000))
        closest_base_station = min(base_stations, key=lambda bs: euclidean_distance(loc, bs.location))
        
        DataCaching(env, x, closest_base_station)
        
        task = GenTask(env, dev_id=x, name_prefix=f"HM_{x}_{pr}", prio=pr, location=loc, ra=relay_agents, bs=base_stations, rcd = Record)
        devices.append(task)
    env.process(trigger_task_removal(env, relay_agents, base_stations))
    env.process(trigger_task_execute(env, base_stations, DataCaching.getDICT()))
    plot_entities(base_stations, relay_agents, devices, 'before')
    env.run(until=600)

    #print("Data Caching Dictionary:", DataCaching.getDICT())
    #print("Task Execution Times:", GenTask.results)
    print("Total size of List is ", len(GenTask.results))
    print("Average Execution Time:", sum(GenTask.results) / len(GenTask.results) if GenTask.results else 0)
    print("Cache Record: cacheHits/cacheMiss")
    print(f" cacheMiss {cacheMiss} and % {cacheMiss *100/ (cacheHits+cacheMiss)}")
    print(f" cacheHits {cacheHits} and % {cacheHits *100/ (cacheHits+cacheMiss)}")
    #print("Execute Dataframe")
    #print(Record1)
    # Merge DataFrames on 'Dev_id' and 'Job_id'
    merged_df = pd.merge(Record, Record1, on=['Dev_id', 'Job_id'])
    merged_df.to_csv('out'+str(dev_num)+'.csv', index=False)
    #print(merged_df)
    #print(merged_df)
    # Calculate Actual Waiting Time
    merged_df['Actual_waiting_time'] = merged_df['waiting_time'] - merged_df['Arraival_time']

    # Calculate Execution Time
    merged_df['Execution_time'] = merged_df['Execution_time_y'] - merged_df['waiting_time']
    # Calculate the average execution time for each Dev_id
    average_execution_time = merged_df.groupby('Dev_id')['Execution_time'].mean().reset_index()

    # Calculate Average Waiting Time per Job
    average_waiting_time_per_job = merged_df['Actual_waiting_time'].mean()

    # Fetch waiting time for each Dev_id
    waiting_time_per_dev = merged_df.groupby('Dev_id')['waiting_time'].sum().reset_index()

    # Fetch average waiting time for each Dev_id
    average_waiting_time_per_dev = merged_df.groupby('Dev_id')['waiting_time'].mean().reset_index()

    # Plot average_waiting_time_per_dev
    plt.figure(figsize=(10, 6))
    plt.bar(average_waiting_time_per_dev['Dev_id'], average_waiting_time_per_dev['waiting_time'], color='skyblue')
    plt.xlabel('Dev_id')
    plt.ylabel('Average Waiting Time')
    plt.title('Average Waiting Time per Device')
    plt.xticks(average_waiting_time_per_dev['Dev_id'])
    #plt.show()

    # Display the DataFrame and results
    #print("DataFrame with Calculations:")
    #print(merged_df)
    print("\nAverage Waiting Time per Job:", average_waiting_time_per_job)
    #print("\nWaiting Time per Dev_id:")
    #print(waiting_time_per_dev)
    avgDataMig = 0
    for x in total_data_migration_time:
               avgDataMig = avgDataMig + x           
    print(f"Avg. data migration time {avgDataMig/len(total_data_migration_time)}")
    print(f"Avg execution time {average_execution_time['Execution_time'].describe()}")
    # Plot waiting_time_per_dev
    plt.figure(figsize=(10, 6))
    plt.bar(waiting_time_per_dev['Dev_id'], waiting_time_per_dev['waiting_time'], color='skyblue')
    plt.xlabel('Dev_id')
    plt.ylabel('Total Waiting Time')
    plt.title('Total Waiting Time per Device')
    plt.xticks(waiting_time_per_dev['Dev_id'])
    #plt.show()
    plot_entities(base_stations, relay_agents, devices,'after')
    # Task Completion Rate (TCR)
    total_tasks = len(Record)
    completed_tasks = len(Record1)
    task_completion_rate = min(100, (completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0

    # Cache Hit Ratio (CHR)
    cache_hit_ratio = (cacheHits / (cacheHits + cacheMiss)) * 100 if (cacheHits + cacheMiss) > 0 else 0

    # Average Task Response Time (ATRT)
    merged_df['Total_Response_Time'] = merged_df['Execution_time'] + merged_df['Actual_waiting_time']
    average_response_time = merged_df['Total_Response_Time'].mean()

    # QoE Score Calculation (0-5 scale)
    # Higher completion rate, lower response time, and higher cache hit ratio improve QoE.
    QoE_Score = (0.4 * (task_completion_rate / 20)) + (0.4 * (cache_hit_ratio / 20)) - (0.2 * (average_response_time / 10))
    QoE_Score = max(0, min(5, QoE_Score))  # Clamping between 0 and 5

    # Print QoE Metrics
    print(f"\n🔹 Task Completion Rate (TCR): {task_completion_rate:.2f}%")
    print(f"🔹 Cache Hit Ratio (CHR): {cache_hit_ratio:.2f}%")
    print(f"🔹 Average Response Time: {average_response_time:.4f} sec")
    print(f"⭐ QoE Score (0-5): {QoE_Score:.2f}")

    # Plot QoE Evaluation
    labels = ['Task Completion Rate (%)', 'Cache Hit Ratio (%)', 'Avg. Response Time (sec)', 'QoE Score']
    values = [task_completion_rate, cache_hit_ratio, average_response_time, QoE_Score * 20]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel("QoE Parameters")
    plt.ylabel("Values")
    plt.title("QoE Evaluation Metrics")
    #plt.show()
    results=[]
    results.append([dev_num,task_completion_rate,cache_hit_ratio,average_response_time,QoE_Score,avgDataMig/len(total_data_migration_time),average_execution_time['Execution_time'].mean()])
    results_df = pd.DataFrame(results, columns=[ 'device_no', 'task_completion_rate','cache_hit_ratio','average_response_time','QoS_score','Datamig_time','exec_time_mean'])
    results_df.to_csv('simulation_results.csv', index=False, mode='a')
    

if __name__ == "__main__":
    main()
