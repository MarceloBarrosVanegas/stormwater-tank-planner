import math
from typing import Dict

class CostCalculator:
    """
    Calculates the detailed cost of stormwater infrastructure based on 
    derivation length and tank volume.
    
    Formulas are based on user specifications:
    - Derivation: Diameter-based step function ($400/m or $1400/m).
    - Tank: Geometric model (Two square chambers, 5m depth) for concrete volume.
    """
    
    # Material Costs
    CONCRETE_UNIT_COST = 150.0       # USD/m3
    STEEL_DENSITY = 120.0            # kg/m3 of concrete
    STEEL_UNIT_COST = 5.0            # USD/kg
    REINFORCED_CONCRETE_COST = CONCRETE_UNIT_COST + (STEEL_DENSITY * STEEL_UNIT_COST) # 750 USD/m3
    
    # Geometric Assumptions
    TANK_DEPTH = 5.0                 # meters
    WALL_THICKNESS = 0.35            # meters
    ROOF_THICKNESS = 0.35            # meters
    BASE_THICKNESS = 0.40            # meters
    
    @staticmethod
    def calculate_derivation_cost(length: float, flow_rate: float) -> float:
        """
        Calculates the cost of the derivation pipe.
        
        Logic:
        - Diameter < 500mm (0.5m) -> $400/m
        - Diameter < 2000mm (2.0m) -> $1400/m
        - Diameter >= 2000mm -> Scaled linearly (Assumption: $2500/m)
        """
        if flow_rate <= 0 or length <= 0:
            return 0.0
            
        # Estimate diameter based on flow rate (Simplified Manning/Velocity approximation)
        # Using a typical V_design = 2.5 m/s. Area = Q / V
        # D = sqrt(4 * Q / (pi * V))
        velocity_design = 2.5 
        area_req = flow_rate / velocity_design
        diameter = math.sqrt((4 * area_req) / math.pi)
        
        if diameter <= 0.5:
            unit_cost = 400.0
        elif diameter <= 2.0:
            unit_cost = 1400.0
        else:
            # Over 2m diameter is huge, assume higher cost
            unit_cost = 2500.0 
            
        return length * unit_cost

    @staticmethod
    def calculate_tank_cost(volume: float) -> float:
        """
        Calculates the construction cost of a reinforced concrete tank.
        
        Geometry: 
        - Two square chambers.
        - Depth = 5m.
        - Calculates concrete volume for Walls, Base, and Roof.
        """
        if volume <= 0:
            return 0.0
            
        # 1. Base Dimensions
        # Total Required Base Area = Volume / Depth
        area_base_internal = volume / CostCalculator.TANK_DEPTH
        
        # Two chambers means the total area is split into two squares.
        # Total Area = 2 * (Width * Width), where Width is the side of one chamber.
        # Width^2 = Area / 2  => Width = sqrt(Area / 2)
        width_internal = math.sqrt(area_base_internal / 2.0)
        
        # Internal Dimensions of the "Two Chamber" footprint:
        # Length side = 2 * Width (two squares side-by-side)
        # Width side = 1 * Width
        length_internal = 2 * width_internal
        
        # 2. Wall Length Calculation
        # External Perimeter = 2 * (Length + Width)
        perimeter_external_walls = 2 * (length_internal + width_internal)
        
        # Internal Dividing Wall (Shared between chambers) = Width
        length_internal_walls = width_internal
        
        total_wall_length = perimeter_external_walls + length_internal_walls
        
        area_walls_vertical = total_wall_length * CostCalculator.TANK_DEPTH
        
        # 3. Concrete Volume Calculation
        # Base Slab
        vol_concrete_base = area_base_internal * CostCalculator.BASE_THICKNESS 
        # Roof Slab
        vol_concrete_roof = area_base_internal * CostCalculator.ROOF_THICKNESS
        # Walls
        vol_concrete_walls = area_walls_vertical * CostCalculator.WALL_THICKNESS
        
        total_concrete_vol = vol_concrete_base + vol_concrete_roof + vol_concrete_walls
        
        # 4. Total Cost
        return total_concrete_vol * CostCalculator.REINFORCED_CONCRETE_COST

    @classmethod
    def calculate_total_construction_cost(cls, length: float, volume: float, flow_rate: float) -> Dict[str, float]:
        """Returns broken down construction costs."""
        cost_deriv = cls.calculate_derivation_cost(length, flow_rate)
        cost_tank = cls.calculate_tank_cost(volume)
        return {
            "total": cost_deriv + cost_tank,
            "derivation": cost_deriv,
            "tank": cost_tank
        }

class FinancialCalculator:
    """
    Centralized Financial Model.
    Handles distinct strategies for calculating flood damage.
    """
    
    def __init__(self, cost_per_m3_flooding: float = 1250.0):
        self.cost_per_m3_flooding = cost_per_m3_flooding
        
    def calculate_damage_volume_based(self, volume_m3: float) -> float:
        """
        Calculates damage purely based on volume.
        Damage = Volume * Cost/m3
        """
        return volume_m3 * self.cost_per_m3_flooding

    def calculate_damage_itzi(self, climada_result: Dict) -> float:
        """
        Retrieves damage calculated by Itzi/Climada integration.
        Returns the specific dollar amount from the physical simulation.
        """
        if climada_result and 'total_damage' in climada_result:
            return float(climada_result['total_damage'])
        return 0.0

if __name__ == "__main__":
    # Quick Test
    print("--- Financial Calculator Test ---")
    fin = FinancialCalculator(cost_per_m3_flooding=100.0)
    
    vol = 5000.0
    dmg_vol = fin.calculate_damage_volume_based(vol)
    print(f"Volume Based Damage ({vol}m3 @ $100): ${dmg_vol:,.2f}")
    
    # Construction
    const_cost = CostCalculator.calculate_total_construction_cost(100, 5000, 2.0)
    print(f"Construction Cost: ${const_cost['total']:,.2f}")
