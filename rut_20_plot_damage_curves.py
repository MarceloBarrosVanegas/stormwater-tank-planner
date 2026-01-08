"""
rut_20_plot_damage_curves.py
Plot JRC flood depth-damage curves for all sectors and save as images.
Creates custom curves for infrastructure and agriculture based on JRC methodology.

Based on JRC Global Flood Depth-Damage Functions (Huizinga et al., 2017)
Source: https://publications.jrc.ec.europa.eu/repository/handle/JRC105688
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config

# Output directory
OUTPUT_DIR = config.ITZI_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# JRC DEPTH-DAMAGE CURVES FOR SOUTH AMERICA
# Extracted from CLIMADA and JRC publications
# =============================================================================

# Depth values (meters) - standard JRC range 0-6m
DEPTHS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

# JRC Damage curves (Mean Damage Ratio from 0 to 1)
# Source: CLIMADA climada_petals.entity.impact_funcs.river_flood

JRC_CURVES = {
    # Residential - South America
    "residential": {
        "mdr": np.array([0.00, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95, 1.00]),
        "color": "#E74C3C",
        "source": "CLIMADA JRC South America",
        "max_damage_usd_m2": 400  # Typical construction value
    },
    
    # Commercial - South America
    "commercial": {
        "mdr": np.array([0.00, 0.15, 0.30, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00]),
        "color": "#3498DB",
        "source": "CLIMADA JRC South America",
        "max_damage_usd_m2": 600
    },
    
    # Industrial - South America
    "industrial": {
        "mdr": np.array([0.00, 0.15, 0.27, 0.37, 0.48, 0.65, 0.80, 0.92, 1.00]),
        "color": "#9B59B6",
        "source": "CLIMADA JRC South America", 
        "max_damage_usd_m2": 350
    },
    
    # Infrastructure/Roads - CUSTOM based on JRC methodology
    # Based on Huizinga et al. (2017) and FEMA road damage functions
    # Lower damage at shallow depths (roads are more resilient)
    "infrastructure": {
        "mdr": np.array([0.00, 0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90]),
        "color": "#27AE60",
        "source": "Custom (JRC/FEMA methodology)",
        "max_damage_usd_m2": 200,  # Road repair cost per m²
        "notes": "Roads and infrastructure are more flood-resistant than buildings"
    },
    
    # Agriculture - CUSTOM based on JRC methodology
    # Crop damage curves - faster initial damage, plateaus earlier
    # Based on JRC agricultural damage functions
    "agriculture": {
        "mdr": np.array([0.00, 0.35, 0.55, 0.70, 0.80, 0.90, 0.95, 1.00, 1.00]),
        "color": "#F39C12",
        "source": "Custom (JRC/FAO methodology)",
        "max_damage_usd_m2": 50,  # Agricultural value per hectare / 10000
        "notes": "Crops are highly vulnerable at low depths"
    }
}


def plot_all_curves_combined(output_dir=None):
    """Plot all damage curves on a single figure."""
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for sector, data in JRC_CURVES.items():
        ax.plot(DEPTHS, data["mdr"] * 100, 
                linewidth=2.5, 
                color=data["color"],
                marker='o',
                markersize=6,
                label=f'{sector.capitalize()} ({data["source"]})')
    
    ax.set_xlabel('Flood Depth (m)', fontsize=12)
    ax.set_ylabel('Damage Ratio (%)', fontsize=12)
    ax.set_title('JRC Flood Depth-Damage Curves for South America\n(Huizinga et al., 2017 + Custom curves)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # Add annotations
    ax.text(0.02, 0.98, 
            'Infrastructure and Agriculture curves derived from\nJRC methodology where CLIMADA lacks implementation',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = output_dir / "damage_curves_all_sectors.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return output_path


def plot_individual_curves(output_dir=None):
    """Plot each damage curve separately with more detail."""
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_paths = []
    
    for sector, data in JRC_CURVES.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Main curve
        ax.fill_between(DEPTHS, 0, data["mdr"] * 100, 
                       color=data["color"], alpha=0.3)
        ax.plot(DEPTHS, data["mdr"] * 100, 
                linewidth=3, 
                color=data["color"],
                marker='o',
                markersize=8)
        
        # Labels on points
        for depth, mdr in zip(DEPTHS, data["mdr"]):
            if depth in [0, 1, 2, 4, 6]:
                ax.annotate(f'{mdr*100:.0f}%', 
                           (depth, mdr*100), 
                           textcoords="offset points", 
                           xytext=(0, 10),
                           ha='center',
                           fontsize=9)
        
        ax.set_xlabel('Flood Depth (m)', fontsize=12)
        ax.set_ylabel('Damage Ratio (%)', fontsize=12)
        ax.set_title(f'Flood Depth-Damage Curve: {sector.upper()}\nSource: {data["source"]}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        
        # Add notes if available
        if "notes" in data:
            ax.text(0.02, 0.02, data["notes"],
                   transform=ax.transAxes, fontsize=9,
                   style='italic', alpha=0.7)
        
        # Add table with values
        table_text = "Depth (m) | Damage (%)\n" + "-"*22 + "\n"
        for d, m in zip(DEPTHS, data["mdr"]):
            table_text += f"   {d:.1f}    |   {m*100:5.1f}\n"
        
        ax.text(0.98, 0.02, table_text,
               transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', horizontalalignment='right',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        output_path = output_dir / f"damage_curve_{sector}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        output_paths.append(output_path)
        plt.close()
    
    return output_paths


def save_curves_as_csv(output_dir=None):
    """Save all curves as CSV for reference."""
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    import pandas as pd
    
    data = {"depth_m": DEPTHS}
    for sector, curve in JRC_CURVES.items():
        data[f"{sector}_mdr"] = curve["mdr"]
        data[f"{sector}_pct"] = curve["mdr"] * 100
    
    df = pd.DataFrame(data)
    output_path = output_dir / "damage_curves_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path


def get_custom_impact_functions():
    """
    Return custom impact functions for sectors not in CLIMADA.
    Can be used to extend rut_19_flood_damage_climada.py
    """
    return {
        "infrastructure": {
            "depths": DEPTHS.tolist(),
            "mdr": JRC_CURVES["infrastructure"]["mdr"].tolist(),
            "source": "Custom (JRC/FEMA methodology)"
        },
        "agriculture": {
            "depths": DEPTHS.tolist(),
            "mdr": JRC_CURVES["agriculture"]["mdr"].tolist(),
            "source": "Custom (JRC/FAO methodology)"
        }
    }


if __name__ == "__main__":
    print("="*60)
    print("PLOTTING JRC FLOOD DAMAGE CURVES")
    print("="*60)
    
    # Plot combined
    combined_path = plot_all_curves_combined()
    
    # Plot individual
    individual_paths = plot_individual_curves()
    
    # Save CSV
    csv_path = save_curves_as_csv()
    
    print("\n" + "="*60)
    print("CURVES GENERATED:")
    print("="*60)
    print(f"  Combined: {combined_path}")
    for p in individual_paths:
        print(f"  Individual: {p}")
    print(f"  Data CSV: {csv_path}")
    print("="*60)
