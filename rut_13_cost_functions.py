import math
from typing import Dict

import config


class CostCalculator:
    """
    Calculates the detailed cost of stormwater infrastructure based on 
    derivation length and tank volume.
    
    Formulas are based on user specifications:
    - Tank: Geometric model (Two square chambers, 5m depth) for concrete volume.
    """
    
    # Material Costs
    CONCRETE_UNIT_COST = 135      # USD/m3
    STEEL_UNIT_COST = 2.16            # USD/kg
    STEEL_WEIGHT_PER_M3_CONCRETE = 90.0    # kg steel per m3 concrete
    
    # Geometric Assumptions             # meters
    WALL_THICKNESS = 0.35            # meters
    ROOF_THICKNESS = 0.35            # meters
    BASE_THICKNESS = 0.40            # meters

    @staticmethod
    def calculate_tank_cost(volume: float) -> float:
        """
        Calculates the construction cost of a reinforced concrete tank.
        
        Geometry: 
        - Two square chambers.

        - Calculates concrete volume for Walls, Base, and Roof.
        """
        if volume <= 0:
            return 0.0
            
        # 1. Internal Dimensions
        # Total Required Internal Base Area = Volume / Depth
        area_base_internal = volume / config.TANK_DEPTH_M
        
        # Two chambers means the total area is split into two squares.
        # Width^2 = Area / 2  => Width = sqrt(Area / 2)
        w_int = math.sqrt(area_base_internal / 2.0)
        l_int = 2 * w_int
        
        # 2. External Dimensions (including wall thickness)
        t = CostCalculator.WALL_THICKNESS
        l_ext = l_int + 2 * t
        w_ext = w_int + 2 * t
        area_base_external = l_ext * w_ext
        
        # 3. Wall Length (Centerline method)
        # Perimeter centerline: 2 * ((L_int + t) + (W_int + t))
        # Plus dividing wall centerline: W_int
        total_wall_length_centerline = 2 * (l_int + t + w_int + t) + w_int
        
        # 4. Concrete Volume Calculation
        # Base and Roof slabs must cover the entire footprint (external area)
        vol_concrete_base = area_base_external * CostCalculator.BASE_THICKNESS 
        vol_concrete_roof = area_base_external * CostCalculator.ROOF_THICKNESS
        
        # Walls volume using centerline length to avoid corner double-counting/missing
        vol_concrete_walls = total_wall_length_centerline * t * config.TANK_DEPTH_M
        
        total_concrete_vol = vol_concrete_base + vol_concrete_roof + vol_concrete_walls
        
        # 5. Total Cost (Concrete + Steel)
        return total_concrete_vol * (
            CostCalculator.CONCRETE_UNIT_COST + 
            CostCalculator.STEEL_WEIGHT_PER_M3_CONCRETE * CostCalculator.STEEL_UNIT_COST
        )


