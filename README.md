# Transmission Fault Detection System

A Python-based system for simulating and detecting faults in power transmission networks using machine learning.

## Project Description

This project implements a power transmission network simulator with real-time fault detection capabilities. It uses the pandapower library to simulate a power grid (specifically the IEEE 30-bus test system) and implements machine learning algorithms to detect and classify different types of faults.

### Implemented Features

1. **Power System Simulation**
   - Uses pandapower for power flow calculations
   - Simulates the IEEE 30-bus test system
   - Real-time monitoring of power system parameters

2. **Fault Simulation**
   - Supports multiple fault types:
     - Line-to-Ground (LG)
     - Line-to-Line (LL)
     - Line-to-Line-to-Ground (LLG)
     - Three-Phase (3P)
   - Configurable fault parameters:
     - Fault impedance
     - Fault location along the line
     - Fault duration

3. **Real-time Monitoring**
   - Tracks key power system parameters:
     - Current magnitude and angle
     - Voltage magnitude and angles
     - Power flow (active and reactive)
     - Current imbalances
   - Maintains historical data for analysis

4. **Fault Detection**
   - Uses Polynomial Regression for fault classification
   - Uses K-Nearest Neighbors (KNN) for fault detection
   - Features extracted from power system measurements
   - Real-time fault prediction capabilities
   - Model training and persistence

### Technical Details

The system consists of several key components:

1. **RealTimeMonitor Class**
   - Collects and processes power system measurements
   - Calculates derived metrics and features
   - Maintains historical data

2. **FaultSimulator Class**
   - Simulates different types of faults
   - Controls fault parameters and timing
   - Manages fault clearing

3. **KNNFaultDetector Class**
   - Implements KNN-based fault detection
   - Uses Polynomial Regression for classification
   - Handles model training and prediction
   - Manages feature extraction and preprocessing


