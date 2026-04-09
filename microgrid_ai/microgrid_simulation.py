import pandas as pd
import numpy as np

class Battery:
    def __init__(self, capacity_kwh=10.0, initial_soc=0.5, efficiency=0.9, max_charge_rate=2.0, max_discharge_rate=2.0):
        self.capacity = capacity_kwh  # kWh
        self.soc = initial_soc  # State of Charge (0.0 to 1.0)
        self.efficiency = efficiency
        self.max_charge_rate = max_charge_rate  # kW
        self.max_discharge_rate = max_discharge_rate  # kW

    def charge(self, power_kw, time_step_h=1.0):
        """Charges the battery with given power (kW). Returns actual power used."""
        power_kw = min(power_kw, self.max_charge_rate)
        energy_to_add = power_kw * time_step_h * self.efficiency
        
        max_energy_can_add = (1.0 - self.soc) * self.capacity
        actual_energy_added = min(energy_to_add, max_energy_can_add)
        
        self.soc += actual_energy_added / self.capacity
        return actual_energy_added / (time_step_h * self.efficiency)

    def discharge(self, power_kw, time_step_h=1.0):
        """Discharges the battery with given power (kW). Returns actual power provided."""
        power_kw = min(power_kw, self.max_discharge_rate)
        energy_to_remove = (power_kw * time_step_h) / self.efficiency
        
        max_energy_can_remove = self.soc * self.capacity
        actual_energy_removed = min(energy_to_remove, max_energy_can_remove)
        
        self.soc -= actual_energy_removed / self.capacity
        return actual_energy_removed * self.efficiency / time_step_h

    def get_soc(self):
        return self.soc * 100 # Percentage

class MicroGridSimulator:
    def __init__(self, battery_capacity=10.0):
        self.battery = Battery(capacity_kwh=battery_capacity)
        self.results = []

    def run_step(self, timestamp, solar_gen, load_demand, controller_action):
        """
        Executes a single time step of the simulation.
        controller_action: Logic from the Controller class
        """
        # 1. Determine Energy Balance
        net_load = load_demand - solar_gen
        
        # 2. Delegate to Controller (this will be handled by controller.py)
        # For simulation purposes, we store the state
        state = {
            'timestamp': timestamp,
            'solar_gen': solar_gen,
            'load_demand': load_demand,
            'net_load': net_load,
            'soc_before': self.battery.get_soc()
        }
        
        # This part will be refined when we connect the controller
        return state

    def record_step(self, step_data):
        self.results.append(step_data)

    def get_results_df(self):
        return pd.DataFrame(self.results)
