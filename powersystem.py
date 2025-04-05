import pandapower as pp
import pandapower.networks as nw
import numpy as np
from datetime import datetime, timedelta
import time
import signal
import sys

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully...")
    running = False
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# 1. Initialize power system
net = nw.case30()
pp.runpp(net)

# 2. System configuration
net.load["p_mw_original"] = net.load.p_mw.copy()
net.load["q_mvar_original"] = net.load.q_mvar.copy()

for tid in net.trafo.index:
    net.trafo.at[tid, "tap_step_percent"] = 1.25
    net.trafo.at[tid, "tap_pos"] = 0

# 3. Data management
class RealTimeMonitor:
    def __init__(self, max_history=3600):  # 1 hour buffer
        self.max_history = max_history  # Initialize the attribute
        self.history = []
        self.previous = {
            'current': {line: 0 for line in net.line.index},
            'voltage': {line: (0, 0) for line in net.line.index}
        }
        self.start_time = datetime.now()
        
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
            
            # Current measurements
            i_ka = net.res_line.i_from_ka.at[line]
            i_deg = self.calculate_current_angle(line)
            
            # Voltage measurements
            v_from = net.res_bus.vm_pu.at[from_bus]
            v_to = net.res_bus.vm_pu.at[to_bus]
            v_deg_from = net.res_bus.va_degree.at[from_bus]
            v_deg_to = net.res_bus.va_degree.at[to_bus]
            
            # Rate of change calculations
            time_diff = 1.0  # Fixed 1-second interval
            di_dt = (i_ka - self.previous['current'][line]) / time_diff
            dv_dt_from = (v_from - self.previous['voltage'][line][0]) / time_diff
            dv_dt_to = (v_to - self.previous['voltage'][line][1]) / time_diff
            
            # Update previous values
            self.previous['current'][line] = i_ka
            self.previous['voltage'][line] = (v_from, v_to)
            
            # Store metrics
            line_metrics[line] = {
                'current': (i_ka, i_deg),
                'voltage_from': (v_from, v_deg_from),
                'voltage_to': (v_to, v_deg_to),
                'di_dt': di_dt,
                'dv_dt_from': dv_dt_from,
                'dv_dt_to': dv_dt_to,
                'impedance': (
                    net.line.r_ohm_per_km.at[line] * net.line.length_km.at[line],
                    net.line.x_ohm_per_km.at[line] * net.line.length_km.at[line]
                ),
                'power_flow': (
                    net.res_line.p_from_mw.at[line],
                    net.res_line.q_from_mvar.at[line]
                )
            }
        self.history.append({
            "timestamp": current_time,
            "lines": line_metrics,
            "system_load": net.load.p_mw.sum()
        })
        
        # Maintain history buffer
        if len(self.history) > self.max_history:
            self.history.pop(0)
# 4. Real-time analysis engine
monitor = RealTimeMonitor()

def live_analysis_loop():
    global running
    while running:
        cycle_start = time.time()
        
        # Update load with realistic pattern
        elapsed = (datetime.now() - monitor.start_time).total_seconds()
        load_factor = 0.9 + 0.1 * np.sin(elapsed/3600 * np.pi)
        net.load["p_mw"] = net.load.p_mw_original * load_factor
        net.load["q_mvar"] = net.load.q_mvar_original * load_factor
        
        # Run power flow
        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("Power flow failed - retrying...")
            continue
        
        # Update metrics
        monitor.update_metrics()
        
        # Live display - MODIFIED TO SHOW ALL LINES
        current = monitor.history[-1]
        print(f"\n{current['timestamp'].strftime('%H:%M:%S')} System Status:")
        print(f"Total Load: {current['system_load']:.1f} MW")
        print("Line Metrics:")
        
        for line in current['lines']:
            metrics = current['lines'][line]
            print(f"Line {line}: "
                  f"I={metrics['current'][0]:.2f}kA ∠{metrics['current'][1]:.1f}°, "
                  f"V_from={metrics['voltage_from'][0]:.3f}pu, "
                  f"dI/dt={metrics['di_dt']:.3f}kA/s")
        
        # Maintain 1-second cycle
        cycle_time = time.time() - cycle_start
        if cycle_time < 1.0:
            time.sleep(1.0 - cycle_time)
# 5. Start analysis
if __name__ == "__main__":
    print("Starting real-time power system monitoring...")
    print("Press Ctrl+C to stop\n")
    
    live_analysis_loop()
    
    # Final report
    print("\nFinal Statistics:")
    print(f"Duration: {datetime.now() - monitor.start_time}")
    print(f"Data Points Collected: {len(monitor.history)}")
    print("Last Recorded State:")
    print(monitor.history[-1])