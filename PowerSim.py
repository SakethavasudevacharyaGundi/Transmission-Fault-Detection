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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
from sklearn.pipeline import Pipeline
import joblib
import os

# Global flags
MANUAL_TEST_LINE = None  # Will be randomly selected
MANUAL_FAULT_TYPE = 'LG'
running = True
TRAINING_MODE = True
MIN_TRAINING_SAMPLES = 200
FAULT_TYPES = ['No Fault', 'LG', 'LL', 'LLG', '3P']
FEATURE_WINDOW_SIZE = 3
MODEL_SAVE_PATH = 'saved_models'

# Create model save directory if it doesn't exist
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

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

            i_ka = net.res_line.i_from_ka.at[line]
            i_deg = self.calculate_current_angle(line)

            v_from = net.res_bus.vm_pu.at[from_bus]
            v_to = net.res_bus.vm_pu.at[to_bus]
            va_from = net.res_bus.va_degree.at[from_bus]
            va_to = net.res_bus.va_degree.at[to_bus]

            time_diff = 1.0
            di_dt = (i_ka - self.previous['current'][line]) / time_diff
            dv_dt_from = (v_from - self.previous['voltage'][line][0]) / time_diff
            dv_dt_to = (v_to - self.previous['voltage'][line][1]) / time_diff

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
        self.fault_history = []
        self.fault_impedance_range = (0.001, 0.1)  # Range of fault impedances in ohms
        self.fault_location_range = (0.0, 1.0)     # Range of fault locations along the line (0-100%)
        self.fault_duration_range = (0.1, 2.0)     # Range of fault durations in seconds
        self.fault_start_time = None
        self.fault_end_time = None

    def create_line_fault(self, line_idx, fault_type):
        try:
            if 'fault' in net and not net.fault.empty:
                net.fault = net.fault.iloc[0:0]

            if fault_type == 'No Fault':
                pp.runpp(net)
                self.last_fault = None
                self.affected_line = None
                self.fault_start_time = None
                self.fault_end_time = None
                return

            from_bus = net.line.from_bus.at[line_idx]
            to_bus = net.line.to_bus.at[line_idx]
            
            # Calculate line length and impedance
            line_length = net.line.length_km.at[line_idx]
            r_ohm = net.line.r_ohm_per_km.at[line_idx] * line_length
            x_ohm = net.line.x_ohm_per_km.at[line_idx] * line_length
            
            # Generate random fault parameters
            fault_impedance = np.random.uniform(*self.fault_impedance_range)
            fault_location = np.random.uniform(*self.fault_location_range)
            fault_duration = np.random.uniform(*self.fault_duration_range)
            
            # Calculate fault location impedance
            r_fault = r_ohm * fault_location
            x_fault = x_ohm * fault_location
            
            # Create fault with calculated parameters
            new_fault = pd.DataFrame([{
                'bus': from_bus,
                'fault_impedance': fault_impedance,
                'fault_type': {
                    'LG': 'lg', 'LL': 'll',
                    'LLG': 'llg', '3P': '3ph'
                }[fault_type],
                'r_fault_ohm': r_fault,
                'x_fault_ohm': x_fault
            }])

            net.fault = new_fault
            pp.runpp(net)
            
            # Record fault details
            current_time = datetime.now()
            fault_details = {
                'type': fault_type,
                'line': line_idx,
                'impedance': fault_impedance,
                'location': fault_location,
                'duration': fault_duration,
                'start_time': current_time,
                'end_time': current_time + timedelta(seconds=fault_duration)
            }
            self.fault_history.append(fault_details)
            self.last_fault = (fault_type, line_idx, current_time)
            self.affected_line = line_idx
            self.fault_start_time = current_time
            self.fault_end_time = current_time + timedelta(seconds=fault_duration)
            
            print(f"\nFault Details:")
            print(f"Type: {fault_type}")
            print(f"Line: {line_idx}")
            print(f"Impedance: {fault_impedance:.3f} ohms")
            print(f"Location: {fault_location*100:.1f}% of line length")
            print(f"Duration: {fault_duration:.1f} seconds")

        except Exception as e:
            print(f"Fault error: {str(e)}")
            if 'fault' in net:
                net.fault = net.fault.iloc[0:0]
            self.last_fault = None
            self.affected_line = None
            self.fault_start_time = None
            self.fault_end_time = None

    def should_clear_fault(self):
        if self.fault_end_time is None:
            return False
        return datetime.now() >= self.fault_end_time

class KNNFaultDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn_detector = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.poly_classifier = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0))
        ])
        self.training_data = []
        self.labels = []
        self.is_trained = False
        self.detection_buffer = deque(maxlen=3)
        self.last_detection = None
        self.min_samples_per_class = 5
        self.load_models()  # Try to load existing models

    def load_models(self):
        try:
            scaler_path = os.path.join(MODEL_SAVE_PATH, 'scaler.joblib')
            knn_path = os.path.join(MODEL_SAVE_PATH, 'knn_detector.joblib')
            poly_path = os.path.join(MODEL_SAVE_PATH, 'poly_classifier.joblib')
            
            if all(os.path.exists(p) for p in [scaler_path, knn_path, poly_path]):
                self.scaler = joblib.load(scaler_path)
                self.knn_detector = joblib.load(knn_path)
                self.poly_classifier = joblib.load(poly_path)
                self.is_trained = True
                print("Loaded existing models successfully")
        except Exception as e:
            print(f"Could not load existing models: {str(e)}")

    def save_models(self):
        try:
            joblib.dump(self.scaler, os.path.join(MODEL_SAVE_PATH, 'scaler.joblib'))
            joblib.dump(self.knn_detector, os.path.join(MODEL_SAVE_PATH, 'knn_detector.joblib'))
            joblib.dump(self.poly_classifier, os.path.join(MODEL_SAVE_PATH, 'poly_classifier.joblib'))
            print("Models saved successfully")
        except Exception as e:
            print(f"Could not save models: {str(e)}")

    def extract_features(self, window):
        features = []
        for snapshot in window:
            for line in snapshot:
                values = snapshot[line]
                # Enhanced feature set
                features.extend([
                    float(values['current_mag']),
                    float(values['voltage_from_mag']),
                    float(values['voltage_to_mag']),
                    float(values['voltage_angle_diff']),
                    float(values['di_dt']),
                    float(values['current_imbalance']),
                    float(values['power_flow_active']),
                    float(values['dv_dt_from']),
                    float(values['dv_dt_to']),
                    # Add derived features
                    float(values['current_mag'] * values['voltage_from_mag']),  # Power factor indicator
                    float(abs(values['di_dt']) * values['voltage_angle_diff']),  # Rate of change indicator
                    float(values['current_imbalance'] * values['power_flow_active'])  # Load imbalance indicator
                ])
        return np.array(features, dtype=np.float64)

    def train(self):
        if len(self.training_data) < MIN_TRAINING_SAMPLES:
            print(f"Not enough training samples ({len(self.training_data)}/{MIN_TRAINING_SAMPLES})")
            return

        X = np.array(self.training_data)
        y = np.array(self.labels)

        # Print class distribution
        print("\nTraining Data Distribution:")
        for ft in FAULT_TYPES:
            count = np.sum(y == ft)
            print(f"{ft}: {count} samples")

        # Ensure minimum samples per class
        class_counts = {ft: np.sum(y == ft) for ft in FAULT_TYPES}
        if any(count < self.min_samples_per_class for count in class_counts.values()):
            print("Insufficient samples for some fault types")
            return

        X_scaled = self.scaler.fit_transform(X)

        # Train KNN for fault detection
        y_binary = (y != 'No Fault').astype(int)
        self.knn_detector.fit(X_scaled, y_binary)

        # Train Polynomial Regression for fault classification
        fault_mask = y_binary == 1
        if sum(fault_mask) > self.min_samples_per_class:
            X_fault = X_scaled[fault_mask]
            y_fault = y[fault_mask]
            self.poly_classifier.fit(X_fault, y_fault)
            self.is_trained = True
            
            # Print training metrics
            knn_score = self.knn_detector.score(X_scaled, y_binary)
            poly_score = self.poly_classifier.score(X_fault, y_fault)
            print(f"\nTraining Results:")
            print(f"KNN Detection Accuracy: {knn_score:.2%}")
            print(f"Polynomial Classification Accuracy: {poly_score:.2%}")
            
            # Save the trained models
            self.save_models()
        else:
            print("Insufficient fault samples for classifier training")
            self.is_trained = False

    def predict(self, features):
        if not self.is_trained:
            return {'fault': False}

        X_scaled = self.scaler.transform([features])
        
        # KNN-based fault detection
        is_fault = self.knn_detector.predict(X_scaled)[0]
        knn_probs = self.knn_detector.predict_proba(X_scaled)[0]
        knn_prob = knn_probs[1] if len(knn_probs) > 1 else knn_probs[0]
        
        # Only add to buffer if confidence is high enough
        if knn_prob > 0.7:  # Confidence threshold
            self.detection_buffer.append(is_fault)
        else:
            self.detection_buffer.append(0)  # Treat low confidence as no fault

        # Clear detection buffer if no detection for 5 seconds
        if self.last_detection and (datetime.now() - self.last_detection).total_seconds() > 5:
            self.detection_buffer.clear()
            self.last_detection = None

        # Require 2 out of 3 consecutive detections for stability
        if sum(self.detection_buffer) >= 2 and not self.last_detection:
            # Polynomial Regression-based fault classification
            fault_type = self.poly_classifier.predict(X_scaled)[0]
            poly_probs = self.poly_classifier.predict_proba(X_scaled)[0]
            
            # Combined confidence score with higher weight on KNN
            confidence = (0.7 * knn_prob + 0.3 * np.max(poly_probs))
            
            self.last_detection = datetime.now()
            return {
                'fault': True,
                'type': fault_type,
                'confidence': confidence,
                'knn_confidence': knn_prob,
                'poly_confidence': np.max(poly_probs)
            }
        
        return {'fault': False}

def live_analysis_loop():
    global running, TRAINING_MODE, MANUAL_TEST_LINE
    monitor = RealTimeMonitor()
    fault_simulator = FaultSimulator()
    detector = KNNFaultDetector()

    manual_fault_triggered = False
    manual_fault_active = False
    training_complete_time = None
    detection_stats = {
        'total_detections': 0,
        'correct_detections': 0,
        'false_positives': 0,
        'missed_faults': 0
    }

    while running:
        start_time = time.time()

        if TRAINING_MODE and np.random.rand() < 0.2:
            # Generate diverse training samples
            fault_type = np.random.choice(FAULT_TYPES[1:])
            line_idx = np.random.choice(net.line.index)
            fault_simulator.create_line_fault(line_idx, fault_type)
        else:
            if not TRAINING_MODE and not manual_fault_triggered:
                if training_complete_time is None:
                    training_complete_time = datetime.now()
                elif (datetime.now() - training_complete_time).total_seconds() > 5:
                    # Randomly select a line for testing
                    MANUAL_TEST_LINE = np.random.choice(net.line.index)
                    print(f"\n=== INJECTING MANUAL {MANUAL_FAULT_TYPE} FAULT ON LINE {MANUAL_TEST_LINE} ===")
                    fault_simulator.create_line_fault(MANUAL_TEST_LINE, MANUAL_FAULT_TYPE)
                    manual_fault_triggered = True
                    manual_fault_active = True

            # Check if current fault should be cleared
            if manual_fault_active and fault_simulator.should_clear_fault():
                print("\n=== CLEARING MANUAL FAULT ===")
                fault_simulator.create_line_fault(MANUAL_TEST_LINE, 'No Fault')
                manual_fault_active = False
                detector.last_detection = None
                detector.detection_buffer.clear()  # Clear detection buffer
                manual_fault_triggered = False

        elapsed = (datetime.now() - monitor.start_time).total_seconds()
        load_factor = 0.9 + 0.1 * np.sin(elapsed / 3600 * np.pi)
        net.load.p_mw = net.load.p_mw_original * load_factor
        net.load.q_mvar = net.load.q_mvar_original * load_factor

        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("Warning: Load flow did not converge")
            continue

        monitor.update_metrics()

        if TRAINING_MODE and len(monitor.feature_window) == FEATURE_WINDOW_SIZE:
            features = detector.extract_features(monitor.feature_window)
            detector.training_data.append(features)
            label = fault_simulator.last_fault[0] if fault_simulator.last_fault else 'No Fault'
            detector.labels.append(label)
            print(f"Training sample added: {label}")

            if len(detector.training_data) >= MIN_TRAINING_SAMPLES:
                detector.train()
                TRAINING_MODE = False
                print("\n=== Training Complete ===")
                print("Switching to detection mode...")

        if not TRAINING_MODE and len(monitor.feature_window) == FEATURE_WINDOW_SIZE:
            features = detector.extract_features(monitor.feature_window)
            prediction = detector.predict(features)

            current_time = datetime.now().strftime('%H:%M:%S')
            system_load = monitor.history[-1]['system_load']
            
            print(f"\n=== Status Update [{current_time}] ===")
            print(f"System Load: {system_load:.1f} MW")
            
            if prediction['fault']:
                detection_stats['total_detections'] += 1
                actual_fault = fault_simulator.last_fault[0] if fault_simulator.last_fault else 'No Fault'
                
                if actual_fault != 'No Fault':
                    if prediction['type'] == actual_fault:
                        detection_stats['correct_detections'] += 1
                        print(f"✓ Correctly detected {actual_fault} fault")
                    else:
                        detection_stats['missed_faults'] += 1
                        print(f"✗ Missed {actual_fault} fault (detected as {prediction['type']})")
                else:
                    detection_stats['false_positives'] += 1
                    print(f"⚠ False positive: detected {prediction['type']} fault")
                
                print(f"KNN Confidence: {prediction['knn_confidence']:.1%}")
                print(f"Polynomial Confidence: {prediction['poly_confidence']:.1%}")
                print(f"Combined Confidence: {prediction['confidence']:.1%}")
            else:
                if fault_simulator.last_fault and fault_simulator.last_fault[0] != 'No Fault':
                    detection_stats['missed_faults'] += 1
                    print(f"✗ Missed {fault_simulator.last_fault[0]} fault")
                else:
                    print("Status: Normal")

            # Print detection statistics periodically
            if detection_stats['total_detections'] > 0 and detection_stats['total_detections'] % 10 == 0:
                print("\n=== Detection Statistics ===")
                print(f"Total Detections: {detection_stats['total_detections']}")
                print(f"Correct Detections: {detection_stats['correct_detections']}")
                print(f"False Positives: {detection_stats['false_positives']}")
                print(f"Missed Faults: {detection_stats['missed_faults']}")
                accuracy = detection_stats['correct_detections'] / detection_stats['total_detections']
                print(f"Detection Accuracy: {accuracy:.1%}")

        elapsed = time.time() - start_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

# Main
if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
    live_analysis_loop()
