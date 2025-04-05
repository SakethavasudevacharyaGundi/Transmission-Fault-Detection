import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import signal
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
from sklearn.decomposition import PCA  # Added for potential visualization

# Global flags
MANUAL_TEST_LINE = 3
EXPECTED_FAULT_TYPE = 'LG'
running = True
TRAINING_MODE = True
MIN_TRAINING_SAMPLES = 50  # Increased training samples
FAULT_TYPES = ['No Fault', 'LG', 'LL', 'LLG', '3P']
FEATURE_WINDOW_SIZE = 3

# Debug counters
class DebugStats:
    def __init__(self):
        self.train_counts = {ft: 0 for ft in FAULT_TYPES}
        self.fault_prob_history = []
        
debug_stats = DebugStats()

# Initialize power system
net = nw.case30()
net.load["p_mw_original"] = net.load.p_mw.copy()
net.load["q_mvar_original"] = net.load.q_mvar.copy()
pp.runpp(net)

class RealTimeMonitor:
    def __init__(self):
        self.history = deque(maxlen=3600)
        self.feature_window = deque(maxlen=FEATURE_WINDOW_SIZE)
        self.previous = {'current': {}, 'voltage': {}}
        self.start_time = datetime.now()
        
        for line in net.line.index:
            self.previous['current'][line] = 0
            self.previous['voltage'][line] = (0, 0)

    def calculate_current_angle(self, line):
        from_bus = net.line.from_bus.at[line]
        p_mw = net.res_line.p_from_mw.at[line]
        q_mvar = net.res_line.q_from_mvar.at[line]
        v_pu = net.res_bus.vm_pu.at[from_bus]
        va_deg = net.res_bus.va_degree.at[from_bus]
        S = (p_mw + 1j * q_mvar) * 1e6
        V = v_pu * np.exp(1j * np.deg2rad(va_deg))
        return np.angle(np.conj(S / V), deg=True)

    def update_metrics(self):
        line_metrics = {}
        current_time = datetime.now()
        
        for line in net.line.index:
            from_bus = net.line.from_bus.at[line]
            to_bus = net.line.to_bus.at[line]
            
            # Current measurements with phase angles
            i_ka = net.res_line.i_from_ka.at[line]
            i_deg = self.calculate_current_angle(line)
            
            # Voltage measurements with angles
            v_from = net.res_bus.vm_pu.at[from_bus]
            v_to = net.res_bus.vm_pu.at[to_bus]
            va_from = net.res_bus.va_degree.at[from_bus]
            va_to = net.res_bus.va_degree.at[to_bus]
            
            # Rate of change calculations
            time_diff = 1.0
            di_dt = (i_ka - self.previous['current'][line]) / time_diff
            dv_dt_from = (v_from - self.previous['voltage'][line][0]) / time_diff
            dv_dt_to = (v_to - self.previous['voltage'][line][1]) / time_diff
            
            # Enhanced features
            line_metrics[line] = {
                'current_mag': i_ka,
                'current_angle': i_deg,
                'voltage_from_mag': v_from,
                'voltage_to_mag': v_to,
                'voltage_angle_diff': abs(va_from - va_to),
                'di_dt': di_dt,
                'dv_dt_from': dv_dt_from,
                'dv_dt_to': dv_dt_to,
                'power_flow_active': net.res_line.p_from_mw.at[line],
                'power_flow_reactive': net.res_line.q_from_mvar.at[line],
                'current_imbalance': abs(i_ka - np.mean([net.res_line.i_from_ka.at[l] for l in net.line.index]))
            }
            
            self.previous['current'][line] = i_ka
            self.previous['voltage'][line] = (v_from, v_to)
        
        self.feature_window.append(line_metrics)
        self.history.append({
            "timestamp": current_time,
            "lines": line_metrics,
            "system_load": net.load.p_mw.sum()
        })

class FaultSimulator:
    def __init__(self):
        self.last_fault = None
        self.affected_line = None
        self.recovery_data = deque(maxlen=2)

    def create_line_fault(self, line_idx, fault_type):
        try:
            if 'fault' in net and not net.fault.empty:
                net.fault = net.fault.iloc[0:0]

            if fault_type == 'No Fault':
                pp.runpp(net)
                self.last_fault = None
                self.affected_line = None
                return

            from_bus = net.line.from_bus.at[line_idx]
            new_fault = pd.DataFrame([{
                'bus': from_bus,
                'fault_impedance': 0.001,
                'fault_type': {
                    'LG': 'lg', 'LL': 'll', 
                    'LLG': 'llg', '3P': '3ph'
                }[fault_type]
            }])

            net.fault = new_fault
            pp.runpp(net)
            self.last_fault = (fault_type, line_idx, datetime.now())
            self.affected_line = line_idx

        except Exception as e:
            print(f"Fault error: {str(e)}")
            if 'fault' in net:
                net.fault = net.fault.iloc[0:0]

class KNNFaultDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.detector = KNeighborsClassifier(n_neighbors=5)  # Reduced neighbors
        self.classifier = None
        self.training_data = []
        self.labels = []
        self.is_trained = False
        self.detection_buffer = deque(maxlen=2)
        self.last_detection = None

    def prepare_features(self, window):
        features = []
        for frame in window:
            frame_features = []
            for line in frame.values():
                # Select more discriminative features
                frame_features.extend([
                    line['current_mag'],
                    line['voltage_from_mag'],
                    line['voltage_to_mag'],
                    line['voltage_angle_diff'],
                    line['di_dt'],
                    line['current_imbalance'],
                    line['power_flow_active']
                ])
            features.extend(frame_features)
        return np.array(features)

    def train_model(self):
        if len(self.training_data) < MIN_TRAINING_SAMPLES:
            print(f"Training: {len(self.training_data)}/{MIN_TRAINING_SAMPLES}")
            return

        X = np.array(self.training_data)
        y = np.array(self.labels)
        
        # Print class distribution
        print("\n--- Training Data Distribution ---")
        for ft in FAULT_TYPES:
            count = np.sum(y[:, 0] == ft)
            print(f"{ft}: {count} samples")
            debug_stats.train_counts[ft] = count
        
        y_detect = (y[:, 0] != 'No Fault').astype(int)
        X_detect = self.scaler.fit_transform(X)
        self.detector.fit(X_detect, y_detect)

        X_class = X[y_detect == 1]
        y_class = y[y_detect == 1][:, 0]
        if len(X_class) > 0:
            self.classifier = KNeighborsClassifier(n_neighbors=min(3, len(X_class)))  # Reduced neighbors
            X_class = self.scaler.transform(X_class)
            self.classifier.fit(X_class, y_class)

        self.is_trained = True
        print(f"\nTrained on {len(X)} samples")
        print(f"Detection accuracy: {self.detector.score(X_detect, y_detect):.2f}")

    def predict(self, features):
        if not self.is_trained or self.classifier is None:
            return {'fault': False}
        
        scaled = self.scaler.transform([features])
        detection = self.detector.predict(scaled)[0]
        self.detection_buffer.append(detection)
        
        if self.last_detection and (datetime.now() - self.last_detection).total_seconds() > 5:
            self.detection_buffer.clear()
            self.last_detection = None

        if sum(self.detection_buffer) >= 2 and not self.last_detection:
            fault_type = self.classifier.predict(scaled)[0]
            probabilities = self.classifier.predict_proba(scaled)[0]
            debug_stats.fault_prob_history.append(probabilities)
            
            print(f"\n[DEBUG] Fault probabilities:")
            for i, ft in enumerate(self.classifier.classes_):
                print(f"{ft}: {probabilities[i]:.2%}")
            
            self.last_detection = datetime.now()
            return {
                'fault': True,
                'type': fault_type,
                'confidence': np.max(probabilities)
            }
        return {'fault': False}

def live_analysis_loop():
    global running, TRAINING_MODE
    monitor = RealTimeMonitor()
    fault_simulator = FaultSimulator()
    detector = KNNFaultDetector()
    
    manual_fault_triggered = False
    manual_fault_active = False
    training_complete_time = None
    
    while running:
        start_time = time.time()
        
        if TRAINING_MODE and np.random.rand() < 0.2:
            fault_type = np.random.choice(FAULT_TYPES[1:])
            line_idx = np.random.choice(net.line.index)
            fault_simulator.create_line_fault(line_idx, fault_type)
        else:
            if not TRAINING_MODE and not manual_fault_triggered:
                if training_complete_time is None:
                    training_complete_time = datetime.now()
                elif (datetime.now() - training_complete_time).total_seconds() > 5:
                    print(f"\n--- INJECTING MANUAL {EXPECTED_FAULT_TYPE} FAULT ON LINE {MANUAL_TEST_LINE} ---")
                    fault_simulator.create_line_fault(MANUAL_TEST_LINE, EXPECTED_FAULT_TYPE)
                    manual_fault_triggered = True
                    manual_fault_active = True
                    clear_fault_time = datetime.now() + timedelta(seconds=2)

            if manual_fault_active and datetime.now() > clear_fault_time:
                print("\n--- CLEARING MANUAL FAULT ---")
                fault_simulator.create_line_fault(MANUAL_TEST_LINE, 'No Fault')
                manual_fault_active = False
                detector.last_detection = None

        # Load variation
        elapsed = (datetime.now() - monitor.start_time).total_seconds()
        load_factor = 0.9 + 0.1 * np.sin(elapsed/3600 * np.pi)
        net.load.p_mw = net.load.p_mw_original * load_factor
        net.load.q_mvar = net.load.q_mvar_original * load_factor
        
        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            continue

        monitor.update_metrics()

        # Training data collection with debug
        if TRAINING_MODE and len(monitor.feature_window) == FEATURE_WINDOW_SIZE:
            features = detector.prepare_features(monitor.feature_window)
            detector.training_data.append(features)
            label = fault_simulator.last_fault[0] if fault_simulator.last_fault else 'No Fault'
            detector.labels.append([label, fault_simulator.last_fault[1] if fault_simulator.last_fault else -1])
            print(f"[DEBUG] Training sample added: {label}")

            if len(detector.training_data) >= MIN_TRAINING_SAMPLES:
                detector.train_model()
                TRAINING_MODE = False

        # Prediction with feature debug
        if not TRAINING_MODE and len(monitor.feature_window) == FEATURE_WINDOW_SIZE:
            features = detector.prepare_features(monitor.feature_window)
            print(f"\n[DEBUG] Current features: {features[:5]}...")  # Show first 5 features
            
            prediction = detector.predict(features)
            
            print(f"\n{datetime.now().strftime('%H:%M:%S')} | Load: {monitor.history[-1]['system_load']:.1f} MW")
            if prediction['fault']:
                status = [
                    f"Detected: {prediction['type']}",
                    f"Expected: {EXPECTED_FAULT_TYPE if manual_fault_active else 'No Fault'}",
                    f"Confidence: {prediction['confidence']:.1%}"
                ]
                print("\n".join(status))
            else:
                print(f"Status: Normal")

        elapsed = time.time() - start_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print("Starting power system monitor...")
    live_analysis_loop()