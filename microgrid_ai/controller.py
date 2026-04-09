import numpy as np
from scipy.optimize import linprog

class MicroGridController:
    def __init__(self, cost_per_kwh=0.15, grid_export_price=0.05):
        self.grid_cost = cost_per_kwh
        self.export_price = grid_export_price

    def decide_action(self, renewable_gen, load_demand, battery_obj):
        """
        Intelligent Energy Management using Linear Optimization:
        Minimize grid electricity cost while maintaining power balance and battery constraints.
        """
        capacity = battery_obj.capacity
        soc = battery_obj.get_soc() / 100.0  # Convert to fraction
        
        # Variables: [battery_charge, battery_discharge, grid_draw, grid_export]
        c = [0, 0, self.grid_cost, -self.export_price]  # Minimize cost
        
        # Equality constraint: power balance
        # renewable + discharge + draw - load - charge - export = 0
        A_eq = [[-1, 1, 1, -1]]
        b_eq = [load_demand - renewable_gen]
        
        # Inequality constraints: SOC bounds
        # For SOC new = soc + (charge - discharge)/capacity
        # 0 <= new <=1
        # So, charge - discharge <= (1 - soc)*capacity
        # discharge - charge <= soc*capacity
        A_ub = [
            [1, -1, 0, 0],  # charge - discharge <= (1 - soc)*capacity
            [-1, 1, 0, 0]   # -charge + discharge <= soc*capacity
        ]
        b_ub = [(1 - soc) * capacity, soc * capacity]
        
        bounds = [(0, None)] * 4
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            battery_charge = res.x[0]
            battery_discharge = res.x[1]
            grid_draw = res.x[2]
            grid_export = res.x[3]
            
            # Apply actions to battery
            charged = battery_obj.charge(battery_charge)
            discharged = battery_obj.discharge(battery_discharge)
            
            battery_action_kw = charged - discharged  # Net action
            
            current_cost = grid_draw * self.grid_cost - grid_export * self.export_price
            
            return {
                'battery_action_kw': battery_action_kw,
                'grid_draw_kw': grid_draw,
                'grid_export_kw': grid_export,
                'current_cost': current_cost,
                'soc_after': battery_obj.get_soc()
            }
        else:
            # Fallback to simple rule-based if optimization fails
            net_power = renewable_gen - load_demand
            battery_action_kw = 0
            grid_draw_kw = 0
            grid_export_kw = 0
            
            if net_power > 0:
                charged = battery_obj.charge(net_power)
                battery_action_kw = charged
                remaining = net_power - charged
                if remaining > 0:
                    grid_export_kw = remaining
            else:
                discharged = battery_obj.discharge(abs(net_power))
                battery_action_kw = -discharged
                remaining = abs(net_power) - discharged
                if remaining > 0:
                    grid_draw_kw = remaining
                    
            current_cost = grid_draw_kw * self.grid_cost - grid_export_kw * self.export_price
            
            return {
                'battery_action_kw': battery_action_kw,
                'grid_draw_kw': grid_draw_kw,
                'grid_export_kw': grid_export_kw,
                'current_cost': current_cost,
                'soc_after': battery_obj.get_soc()
            }
    def rl_decide_action(self, state):
        """Placeholder for RL logic."""
        # state = [solar_forecast, current_load, current_soc, time_of_day]
        pass
