import pandapower as pp
import pandapower.networks as nw
import pandapower.shortcircuit as sc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import signal
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque
import copy

# Configuration
MANUAL_TEST_LINE = 3
EXPECTED_FAULT_TYPE = 'LG'  
TRAINING_MODE = True
MIN_TRAINING_SAMPLES = 50
FEATURE_WINDOW_SIZE = 3
FAULT_TYPES = ['No Fault', 'LG', 'LL', '3P']
running = True

# Initialize network and add required parameters
net = nw.case30()
pp.convert_format(net)

# Save original load values for fluctuation simulation
net.load['p_mw_original'] = net.load.p_mw.copy()
net.load['q_mvar_original'] = net.load.q_mvar.copy()

def sanitize_network_for_short_circuit(net):
    """
    Enhanced sanitization to handle lines, transformers, generators, and external grids.
    """
    # Zero-sequence parameters for lines
    for col in ['r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km']:
        if col not in net.line.columns:
            base_col = col.replace('0_', '')
            if base_col in net.line.columns:
                net.line[col] = 3 * net.line[base_col]
            else:
                net.line[col] = 0.1
        net.line[col] = net.line[col].fillna(0.1).replace(0, 0.1)

    # Essential line parameters
    essential_line_columns = {
        'r_ohm_per_km': 0.01,
        'x_ohm_per_km': 0.05,
        'length_km': 1.0,
        'from_bus': 0,
        'to_bus': 0
    }
    for col, default_val in essential_line_columns.items():
        if col not in net.line.columns:
            net.line[col] = default_val
        net.line[col] = net.line[col].fillna(default_val).replace(0, default_val)

    # Optional sanitization of type/std_type
    net.line['type'] = net.line.get('type', 'ol').fillna('ol')
    net.line['std_type'] = net.line.get('std_type', None).fillna(None)

    # Sanitize transformers
    if 'trafo' in net and not net.trafo.empty:
        trafo_zero_seq = ['vk0_percent', 'vkr0_percent', 'mag0_percent']
        for param in trafo_zero_seq:
            if param not in net.trafo.columns:
                net.trafo[param] = 0.0
            net.trafo[param] = net.trafo[param].fillna(0.0)
        
        essential_trafo = ['vk_percent', 'vkr_percent', 'sn_mva', 'vn_hv_kv', 'vn_lv_kv']
        defaults = [10.0, 0.5, 40.0, 110.0, 20.0]
        for param, default in zip(essential_trafo, defaults):
            if param not in net.trafo.columns:
                net.trafo[param] = default
            net.trafo[param] = net.trafo[param].fillna(default)

    # Sanitize generators
    if 'gen' in net and not net.gen.empty:
        essential_gen = {
            'pg_percent': 100.0,
            'vn_kv': net.bus.loc[net.gen.bus, 'vn_kv'].values,
            'xdss': 20.0,
            'rdss': 0.1
        }
        for param, default in essential_gen.items():
            if param not in net.gen.columns:
                net.gen[param] = default
            net.gen[param] = net.gen[param].fillna(default)

    # Sanitize external grid
    if 'ext_grid' in net and not net.ext_grid.empty:
        essential_extgrid = {
            's_sc_max_mva': 10000.0,
            'rx_max': 0.1,
            'x0x_max': 1.0,
            'r0x0_max': 0.1
        }
        for param, default in essential_extgrid.items():
            if param not in net.ext_grid.columns:
                net.ext_grid[param] = default
            net.ext_grid[param] = net.ext_grid[param].fillna(default)

    # Final fill for any remaining NaNs in lines
    net.line = net.line.fillna(0.1)
    invalid_lines = net.line.isnull().any(axis=1)
    if invalid_lines.any():
        print(f"Dropping {invalid_lines.sum()} invalid lines")
        net.line = net.line[~invalid_lines]

def add_network_parameters():
    # Line zero-sequence params (fallback)
    for col in ['r0_ohm_per_km', 'x0_ohm_per_km', 'c0_nf_per_km']:
        if col not in net.line.columns:
            base_col = col.replace('0', '')
            net.line[col] = 3 * net.line[base_col] if base_col in net.line.columns else 0.1

    # Generator parameters are handled in the sanitizer
    sanitize_network_for_short_circuit(net)

class RealTimeMonitor:
    def __init__(self):
        self.feature_window = deque(maxlen=FEATURE_WINDOW_SIZE)
        self.previous = {'current': {}, 'voltage': {}}
        for line in net.line.index:
            self.previous['current'][line] = 0
            self.previous['voltage'][line] = (0, 0)
        self.start_time = datetime.now()

    def update_metrics(self):
        line_metrics = {}
        for line in net.line.index:
            from_bus = net.line.from_bus.at[line]
            to_bus = net.line.to_bus.at[line]

            i_ka = net.res_line.i_from_ka.at[line]
            v_from = net.res_bus.vm_pu.at[from_bus]
            v_to = net.res_bus.vm_pu.at[to_bus]
            va_from = net.res_bus.va_degree.at[from_bus]
            va_to = net.res_bus.va_degree.at[to_bus]

            di_dt = i_ka - self.previous['current'][line]
            dv_dt_from = v_from - self.previous['voltage'][line][0]
            dv_dt_to = v_to - self.previous['voltage'][line][1]

            line_metrics[line] = {
                'current_mag': i_ka,
                'voltage_from_mag': v_from,
                'voltage_to_mag': v_to,
                'voltage_angle_diff': abs(va_from - va_to),
                'di_dt': di_dt,
                'dv_dt_from': dv_dt_from,
                'dv_dt_to': dv_dt_to,
                'power_flow_active': net.res_line.p_from_mw.at[line],
                'current_imbalance': abs(i_ka - np.mean(net.res_line.i_from_ka))
            }

            self.previous['current'][line] = i_ka
            self.previous['voltage'][line] = (v_from, v_to)

        self.feature_window.append(line_metrics)

class FaultSimulator:
    def __init__(self):
        self.last_fault = None

    def apply_fault(self, line_idx, fault_type):
        global net
        if fault_type == 'No Fault':
            try:
                pp.runpp(net)
            except pp.LoadflowNotConverged:
                print("Load flow did not converge during no-fault case.")
            self.last_fault = None
            return

        if line_idx not in net.line.index:
            return

        from_bus = net.line.from_bus.at[line_idx]
        original_net = copy.deepcopy(net)

        try:
            add_network_parameters()

            # PATCH: Ensure all zero-sequence and required fault params are present
            if 's_sc_max_mva' not in net.ext_grid:
                net.ext_grid["s_sc_max_mva"] = 10000.0
            if 'rx_max' not in net.ext_grid:
                net.ext_grid["rx_max"] = 0.1
            if 'x0x_max' not in net.ext_grid:
                net.ext_grid["x0x_max"] = 1.0
            if 'r0x0_max' not in net.ext_grid:
                net.ext_grid["r0x0_max"] = 0.1

            # PATCH: Detect NaNs in fault-relevant dataframes before sc
            fault_inputs = {
                "line": net.line,
                "trafo": net.trafo if not net.trafo.empty else pd.DataFrame(),
                "gen": net.gen if not net.gen.empty else pd.DataFrame(),
                "ext_grid": net.ext_grid
            }

            for key, df in fault_inputs.items():
                if df.isnull().values.any():
                    nan_cols = df.columns[df.isnull().any()].tolist()
                    print(f"⚠️ NaNs found in {key} columns: {nan_cols}")
                    raise ValueError(f"NaNs in {key} before short-circuit calculation")

            # Run the fault
            if fault_type == 'LG':
                sc.calc_sc(net, fault="1ph", bus=from_bus, r_fault_ohm=0.001, x_fault_ohm=0.001)
            elif fault_type == 'LL':
                sc.calc_sc(net, fault="2ph", bus=from_bus, r_fault_ohm=0.001, x_fault_ohm=0.001)
            elif fault_type == '3P':
                sc.calc_sc(net, fault="3ph", bus=from_bus, r_fault_ohm=0.001, x_fault_ohm=0.001)

            self.last_fault = (fault_type, line_idx)

        except Exception as e:
            print(f"Short-circuit calculation failed: {e}")
            print(f"Restoring network to previous state.")
            net = copy.deepcopy(original_net)
            self.last_fault = None


class KNNFaultDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.knn_detector = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier = None
        self.training_data = []
        self.labels = []
        self.is_trained = False

    def extract_features(self, window):
        features = []
        for snapshot in window:
            for line in snapshot:
                values = snapshot[line]
                try:
                    features.extend([
                        float(values['current_mag']),
                        float(values['voltage_from_mag']),
                        float(values['voltage_to_mag']),
                        float(values['voltage_angle_diff']),
                        float(values['di_dt']),
                        float(values['current_imbalance']),
                        float(values['power_flow_active'])
                    ])
                except Exception as e:
                    print(f"Bad feature value: {values} | Error: {e}")
        return np.array(features, dtype=np.float64)

    def train(self):
        if len(self.training_data) < MIN_TRAINING_SAMPLES:
            print(f"Not enough training samples ({len(self.training_data)}/{MIN_TRAINING_SAMPLES})")
            return

        X = np.array(self.training_data)
        y = np.array(self.labels)
        X_scaled = self.scaler.fit_transform(X)

        y_binary = (y != 'No Fault').astype(int)
        self.knn_detector.fit(X_scaled, y_binary)

        fault_mask = y_binary == 1
        if sum(fault_mask) > 5:
            self.knn_classifier = KNeighborsClassifier(n_neighbors=3)
            self.knn_classifier.fit(X_scaled[fault_mask], y[fault_mask])
            self.is_trained = True
            print(f"Trained classifier with {sum(fault_mask)} fault samples")
        else:
            print("Insufficient fault samples for classifier training")
            self.is_trained = False

    def predict(self, features):
        if not self.is_trained:
            return {'fault': False}

        X_scaled = self.scaler.transform([features])
        is_fault = self.knn_detector.predict(X_scaled)[0]

        if is_fault and self.knn_classifier:
            fault_type = self.knn_classifier.predict(X_scaled)[0]
            prob = self.knn_classifier.predict_proba(X_scaled).max()
            return {'fault': True, 'type': fault_type, 'confidence': prob}

        return {'fault': False}

def main_loop():
    global running, TRAINING_MODE
    monitor = RealTimeMonitor()
    simulator = FaultSimulator()
    detector = KNNFaultDetector()
    training_complete_time = None
    manual_fault_triggered = False

    while running:
        start = time.time()

        elapsed = (datetime.now() - monitor.start_time).total_seconds()
        net.load.p_mw = net.load.p_mw_original * (0.9 + 0.1 * np.sin(elapsed / 3600 * np.pi))
        net.load.q_mvar = net.load.q_mvar_original * (0.9 + 0.1 * np.sin(elapsed / 3600 * np.pi))

        if TRAINING_MODE:
            if np.random.rand() < 0.3:
                f_type = np.random.choice(FAULT_TYPES[1:])
                line = np.random.choice(net.line.index)
                simulator.apply_fault(line, f_type)
            else:
                simulator.apply_fault(None, 'No Fault')
        else:
            if not manual_fault_triggered:
                if training_complete_time is None:
                    training_complete_time = datetime.now()
                elif (datetime.now() - training_complete_time).total_seconds() > 3:
                    print(f"\n=== Injecting Manual Fault {EXPECTED_FAULT_TYPE} on Line {MANUAL_TEST_LINE} ===")
                    simulator.apply_fault(MANUAL_TEST_LINE, EXPECTED_FAULT_TYPE)
                    manual_fault_triggered = True
                    fault_clear_time = datetime.now() + timedelta(seconds=3)

            if manual_fault_triggered and datetime.now() > fault_clear_time:
                print("\n=== Clearing Manual Fault ===")
                simulator.apply_fault(MANUAL_TEST_LINE, 'No Fault')
                manual_fault_triggered = False

        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("Load flow didn't converge")
            continue

        monitor.update_metrics()

        if len(monitor.feature_window) == FEATURE_WINDOW_SIZE:
            features = detector.extract_features(monitor.feature_window)

            if TRAINING_MODE:
                label = simulator.last_fault[0] if simulator.last_fault else 'No Fault'
                detector.training_data.append(features)
                detector.labels.append(label)
                print(f"Training sample added: {label}")

                if len(detector.training_data) >= MIN_TRAINING_SAMPLES:
                    detector.train()
                    TRAINING_MODE = False
                    print("Training complete. Switching to detection mode.")
            else:
                result = detector.predict(features)
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Load = {net.load.p_mw.sum():.2f} MW")
                if result['fault']:
                    print(f"FAULT DETECTED! Type: {result['type']} | Confidence: {result['confidence']:.2%}")
                else:
                    print("Status: Normal")

        time.sleep(max(0.5 - (time.time() - start), 0))

def signal_handler(sig, frame):
    global running
    print("\nExiting safely...")
    running = False
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print("Starting Real-Time Power System Monitor with Fault Detection")
    main_loop()
