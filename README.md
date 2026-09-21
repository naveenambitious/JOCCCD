# JOCCCD: Joint Optimization of Communication, Caching, and Computing Decisions

This repository provides the implementation and simulation environment for the **JOCCCD framework**, which jointly optimizes communication channels, data caching, and computation offloading in Fog Internet of Things (FIoT) systems to improve Quality of Experience (QoE).

---

## 📂 Repository Structure

- `Joint2wRes.py`  
  Main simulation file implementing the complete JOCCCD framework.

- `JointwRes_nocache.py`  
  Simulation script for evaluating the system **without caching**.

- `JointwRes_nondynamic.py`  
  Simulation script for evaluating the system **without dynamic bandwidth allocation**.

---

## ⚙️ Simulation Overview

The simulation framework is developed in **Python** and models a dynamic Fog IoT environment with the following characteristics:

- **Grid Size:** 1000 × 1000 units  
- **IoT Devices:** Randomly distributed and mobile  
- **Mobility Model:** Random movement with speeds ranging from **1 to 5 km/h**  
- **Network Architecture:**
  - IoT devices connect to **relay agents**
  - Relay agents act as **intermediate nodes (store-and-forward mechanism)**
  - Relay agents are connected to the **nearest fog nodes**
- **Dynamic Behavior:**
  - Devices frequently connect/disconnect due to mobility
  - Tasks are forwarded to the nearest or most suitable fog node
  - Cached data is utilized to reduce **data migration time**

The simulation parameters are configured based on the setup described in **Table~\ref{simulation_setup}** of the manuscript.

---

## 🚀 Features

- Joint optimization of:
  - Communication bandwidth allocation  
  - Data caching strategy  
  - Task offloading decisions  

- **Dynamic bandwidth allocation** based on traffic conditions  
- **Cache-aware scheduling** to improve hit ratio  
- Support for **heterogeneous workloads and device mobility**

---

## 📊 Output

Each simulation run generates:

- `simulation_results.csv` → aggregated performance metrics  
- `out_<devices>.csv` → detailed results for each device configuration (e.g., `out2000.csv`)  

---

## 📈 Performance Metrics

The framework evaluates system performance using:

- Job Completion Time (ms)  
- Cache Hit Ratio (%)  
- Cache Miss Ratio (%)  
- Average Response Time  
- QoE Score (0–5 scale)  

---

## 🔬 Experimental Analysis

The repository supports:

- **Baseline comparison** with existing methods  
- **Ablation studies**:
  - No caching scenario  
  - No dynamic bandwidth allocation scenario  

These experiments demonstrate the contribution of each optimization component in improving QoE and resource utilization.

---

## ▶️ How to Run

Run simulations using:

```bash
python Joint2wRes.py <number_of_devices> caching
