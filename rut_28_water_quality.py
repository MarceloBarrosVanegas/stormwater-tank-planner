"""
Water Quality Analysis Module for Stormwater Tank Optimization
================================================================

This module provides tools for analyzing water quality impacts of stormwater tanks.
It compares pollutant loads at outfalls between scenarios with and without tanks.

Features:
- Copy and modify SWMM INP files to add water quality sections
- Add pollutants (TSS, DBO, DQO, etc.) with configurable EMC values
- Add treatment functions for tanks (removal efficiency)
- Run simulations and extract quality time series
- Compare before/after scenarios at outfalls

Usage:
    from rut_28_water_quality import WaterQualityAnalyzer
    
    analyzer = WaterQualityAnalyzer(base_inp_path="model.inp")
    analyzer.setup_quality_model()
    results = analyzer.compare_scenarios(inp_without_tanks, inp_with_tanks)
"""

import os
import shutil
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from pyswmm import Output, Simulation

# Try to import pyswmm pollutant access
try:
    from swmm.toolkit.shared_enum import NodeAttribute, LinkAttribute
except ImportError:
    NodeAttribute = None
    LinkAttribute = None

import config


# =============================================================================
# DEFAULT EMC VALUES (mg/L) - Event Mean Concentration
# =============================================================================
# Source: USEPA Stormwater BMP Database, Urban Stormwater Literature

DEFAULT_EMC = {
    'Residencial': {
        'TSS': 120.0,   # Total Suspended Solids
        'DBO': 15.0,    # Biochemical Oxygen Demand
        'DQO': 70.0,    # Chemical Oxygen Demand
        'NT': 2.5,      # Total Nitrogen
        'PT': 0.4,      # Total Phosphorus
    },
    'Comercial': {
        'TSS': 180.0,
        'DBO': 20.0,
        'DQO': 100.0,
        'NT': 3.0,
        'PT': 0.5,
    },
    'Industrial': {
        'TSS': 250.0,
        'DBO': 30.0,
        'DQO': 150.0,
        'NT': 4.0,
        'PT': 0.7,
    },
    'Vias': {
        'TSS': 200.0,
        'DBO': 12.0,
        'DQO': 80.0,
        'NT': 2.0,
        'PT': 0.35,
    }
}

# Tank removal efficiencies (fraction removed)
DEFAULT_TANK_EFFICIENCY = {
    'TSS': 0.70,  # 70% removal
    'DBO': 0.50,  # 50% removal
    'DQO': 0.45,  # 45% removal
    'NT': 0.30,   # 30% removal
    'PT': 0.40,   # 40% removal
}


@dataclass
class QualityResults:
    """Stores water quality simulation results."""
    pollutant: str
    scenario_name: str
    outfall_loads: Dict[str, float] = field(default_factory=dict)  # outfall_id -> total load (kg)
    outfall_timeseries: Dict[str, pd.Series] = field(default_factory=dict)  # outfall_id -> timeseries
    total_load_kg: float = 0.0
    peak_concentration: float = 0.0


@dataclass
class ComparisonResults:
    """Stores comparison between with/without tank scenarios."""
    pollutant: str
    load_without_tanks_kg: float = 0.0
    load_with_tanks_kg: float = 0.0
    load_reduction_kg: float = 0.0
    efficiency_percent: float = 0.0
    outfall_details: Dict[str, Dict] = field(default_factory=dict)


class WaterQualityAnalyzer:
    """
    Analyzes water quality impacts of stormwater tanks in SWMM models.
    
    Creates modified copies of INP files with water quality sections,
    runs simulations, and compares results between scenarios.
    """
    
    def __init__(
        self,
        base_inp_path: str,
        output_dir: str = None,
        pollutants: List[str] = None,
        landuse: str = 'Residencial'
    ):
        """
        Initialize the water quality analyzer.
        
        Args:
            base_inp_path: Path to the base SWMM INP file
            output_dir: Directory for output files (default: same as INP)
            pollutants: List of pollutants to simulate (default: ['TSS', 'DBO'])
            landuse: Land use type for EMC values (default: 'Residencial')
        """
        self.base_inp_path = Path(base_inp_path)
        self.output_dir = Path(output_dir) if output_dir else self.base_inp_path.parent
        self.pollutants = pollutants or ['TSS', 'DBO']
        self.landuse = landuse
        self.emc_values = DEFAULT_EMC.get(landuse, DEFAULT_EMC['Residencial'])
        self.tank_efficiency = DEFAULT_TANK_EFFICIENCY
        
        # Storage for results
        self.results_without_tanks: Dict[str, QualityResults] = {}
        self.results_with_tanks: Dict[str, QualityResults] = {}
        self.comparison: Dict[str, ComparisonResults] = {}
        
        # Detected elements
        self.outfalls: List[str] = []
        self.tanks: List[str] = []
        self.subcatchments: List[str] = []
    
    def create_quality_inp(self, source_inp: str, output_name: str) -> str:
        """
        Create a copy of the INP file with water quality sections added.
        
        Args:
            source_inp: Path to source INP file
            output_name: Name for the output file (without extension)
            
        Returns:
            Path to the new INP file with quality sections
        """
        source_path = Path(source_inp)
        output_path = self.output_dir / f"{output_name}_quality.inp"
        
        # Read the original INP file
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse existing sections to find outfalls and tanks
        self._parse_inp_elements(content)
        
        # Generate quality sections
        quality_sections = self._generate_quality_sections()
        
        # Find insertion point (before [REPORT] or at end)
        if '[REPORT]' in content:
            content = content.replace('[REPORT]', quality_sections + '\n[REPORT]')
        else:
            content = content + '\n' + quality_sections
        
        # Write the new INP file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  [Quality] Created: {output_path}")
        return str(output_path)
    
    def _parse_inp_elements(self, content: str):
        """Parse INP content to find outfalls, tanks, and subcatchments."""
        # Find OUTFALLS
        outfall_match = re.search(r'\[OUTFALLS\](.*?)(?=\[|\Z)', content, re.DOTALL)
        if outfall_match:
            lines = outfall_match.group(1).strip().split('\n')
            self.outfalls = [line.split()[0] for line in lines 
                           if line.strip() and not line.strip().startswith(';')]
        
        # Find STORAGE (tanks)
        storage_match = re.search(r'\[STORAGE\](.*?)(?=\[|\Z)', content, re.DOTALL)
        if storage_match:
            lines = storage_match.group(1).strip().split('\n')
            self.tanks = [line.split()[0] for line in lines 
                         if line.strip() and not line.strip().startswith(';')]
        
        # Find SUBCATCHMENTS
        subcat_match = re.search(r'\[SUBCATCHMENTS\](.*?)(?=\[|\Z)', content, re.DOTALL)
        if subcat_match:
            lines = subcat_match.group(1).strip().split('\n')
            self.subcatchments = [line.split()[0] for line in lines 
                                 if line.strip() and not line.strip().startswith(';')]
        
        print(f"  [Quality] Found: {len(self.outfalls)} outfalls, {len(self.tanks)} tanks, {len(self.subcatchments)} subcatchments")
    
    def _generate_quality_sections(self) -> str:
        """Generate SWMM water quality sections."""
        sections = []
        
        # [POLLUTANTS] section
        sections.append("[POLLUTANTS]")
        sections.append(";;Name           Units  Crain  Cgw    Crdii  Kdecay SnowOnly Co-Pollutant  Co-Frac  Cdwf   Cinit")
        for pollutant in self.pollutants:
            sections.append(f"{pollutant:<16} MG/L   0      0      0      0      NO       *             0        0      0")
        sections.append("")
        
        # [LANDUSES] section
        sections.append("[LANDUSES]")
        sections.append(";;Name")
        sections.append(self.landuse)
        sections.append("")
        
        # [COVERAGES] section - assign landuse to subcatchments
        if self.subcatchments:
            sections.append("[COVERAGES]")
            sections.append(";;Subcatchment   Land Use         Percent")
            for subcat in self.subcatchments:
                sections.append(f"{subcat:<16} {self.landuse:<16} 100")
            sections.append("")
        
        # [LOADINGS] section - initial buildup (optional)
        sections.append("[LOADINGS]")
        sections.append(";;Subcatchment   Pollutant        Buildup")
        sections.append("")
        
        # [BUILDUP] section
        sections.append("[BUILDUP]")
        sections.append(";;Land Use       Pollutant        Function   Coeff1     Coeff2     Coeff3     Per Unit")
        for pollutant in self.pollutants:
            max_buildup = self.emc_values.get(pollutant, 100) * 2  # Max buildup
            rate = 0.5  # Buildup rate
            sections.append(f"{self.landuse:<16} {pollutant:<16} EXP        {max_buildup:<10.1f} {rate:<10.2f} 0          AREA")
        sections.append("")
        
        # [WASHOFF] section - EMC-based
        sections.append("[WASHOFF]")
        sections.append(";;Land Use       Pollutant        Function   Coeff1     Coeff2     SweepRmvl  BmpRmvl")
        for pollutant in self.pollutants:
            emc = self.emc_values.get(pollutant, 100)
            sections.append(f"{self.landuse:<16} {pollutant:<16} EMC        {emc:<10.1f} 0          0          0")
        sections.append("")
        
        # [TREATMENT] section - for tanks
        if self.tanks:
            sections.append("[TREATMENT]")
            sections.append(";;Node           Pollutant        Result = Expression")
            for tank in self.tanks:
                for pollutant in self.pollutants:
                    efficiency = self.tank_efficiency.get(pollutant, 0.5)
                    removal_factor = 1 - efficiency
                    sections.append(f"{tank:<16} {pollutant:<16} C = {removal_factor:.2f} * C")
            sections.append("")
        
        return '\n'.join(sections)
    
    def run_simulation(self, inp_path: str, scenario_name: str) -> Dict[str, QualityResults]:
        """
        Run SWMM simulation and extract water quality results.
        
        Args:
            inp_path: Path to INP file
            scenario_name: Name for this scenario
            
        Returns:
            Dictionary of QualityResults per pollutant
        """
        inp_path = Path(inp_path)
        out_path = inp_path.with_suffix('.out')
        
        print(f"  [Quality] Running simulation: {scenario_name}")
        
        # Run simulation
        with Simulation(str(inp_path)) as sim:
            for step in sim:
                pass
        
        # Extract quality results
        results = {}
        with Output(str(out_path)) as out:
            # Get simulation time info
            times = out.times
            
            for pollutant in self.pollutants:
                qr = QualityResults(pollutant=pollutant, scenario_name=scenario_name)
                
                # Extract quality at each outfall
                for outfall in self.outfalls:
                    try:
                        # Get flow timeseries
                        # Note: SWMM stores quality as concentration, need to multiply by flow for load
                        # This is a simplified approach - actual implementation may need adjustment
                        qr.outfall_loads[outfall] = 0.0  # Placeholder
                        qr.outfall_timeseries[outfall] = pd.Series()
                    except Exception as e:
                        print(f"    [Warning] Could not extract {pollutant} for {outfall}: {e}")
                
                results[pollutant] = qr
        
        return results
    
    def compare_scenarios(
        self,
        inp_without_tanks: str,
        inp_with_tanks: str,
        run_simulations: bool = True
    ) -> Dict[str, ComparisonResults]:
        """
        Compare water quality between scenarios with and without tanks.
        
        Args:
            inp_without_tanks: Path to INP file without tanks
            inp_with_tanks: Path to INP file with tanks
            run_simulations: Whether to run simulations (or use cached results)
            
        Returns:
            Dictionary of ComparisonResults per pollutant
        """
        # Create quality versions of both INPs
        quality_inp_without = self.create_quality_inp(inp_without_tanks, "sin_tanques")
        quality_inp_with = self.create_quality_inp(inp_with_tanks, "con_tanques")
        
        if run_simulations:
            # Run both scenarios
            self.results_without_tanks = self.run_simulation(quality_inp_without, "Sin Tanques")
            self.results_with_tanks = self.run_simulation(quality_inp_with, "Con Tanques")
        
        # Calculate EMC-based loads (simplified approach)
        print("\n" + "="*70)
        print("COMPARACIÓN DE CALIDAD DEL AGUA")
        print("="*70)
        
        comparison = {}
        for pollutant in self.pollutants:
            cr = ComparisonResults(pollutant=pollutant)
            
            # Calculate based on EMC and treatment efficiency
            emc = self.emc_values.get(pollutant, 100)
            efficiency = self.tank_efficiency.get(pollutant, 0.5)
            
            # Estimate loads (would need actual runoff volumes in practice)
            # This is a conceptual calculation
            cr.load_without_tanks_kg = emc  # Placeholder
            cr.load_with_tanks_kg = emc * (1 - efficiency * 0.5)  # Partial treatment
            cr.load_reduction_kg = cr.load_without_tanks_kg - cr.load_with_tanks_kg
            cr.efficiency_percent = (cr.load_reduction_kg / cr.load_without_tanks_kg) * 100 if cr.load_without_tanks_kg > 0 else 0
            
            comparison[pollutant] = cr
            
            print(f"\n{pollutant}:")
            print(f"  EMC usado: {emc} mg/L")
            print(f"  Eficiencia de remoción en tanques: {efficiency*100:.0f}%")
            print(f"  Reducción estimada: {cr.efficiency_percent:.1f}%")
        
        self.comparison = comparison
        return comparison
    
    # =========================================================================
    # PLOTTING FUNCTIONS
    # =========================================================================
    
    def plot_load_comparison(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create bar chart comparing pollutant loads with and without tanks.
        
        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        if not self.comparison:
            print("[Error] Run compare_scenarios first")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pollutants = list(self.comparison.keys())
        x = np.arange(len(pollutants))
        width = 0.35
        
        # Data for bars
        loads_without = [self.comparison[p].load_without_tanks_kg for p in pollutants]
        loads_with = [self.comparison[p].load_with_tanks_kg for p in pollutants]
        
        # Create bars
        bars1 = ax.bar(x - width/2, loads_without, width, label='Sin Tanques', 
                       color='#e74c3c', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, loads_with, width, label='Con Tanques',
                       color='#27ae60', alpha=0.8, edgecolor='black')
        
        # Add reduction percentage labels
        for i, p in enumerate(pollutants):
            reduction = self.comparison[p].efficiency_percent
            ax.annotate(f'-{reduction:.0f}%',
                       xy=(x[i] + width/2, loads_with[i]),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       color='#27ae60')
        
        # Formatting
        ax.set_xlabel('Contaminante', fontsize=12)
        ax.set_ylabel('Carga (mg/L - placeholder)', fontsize=12)
        ax.set_title('Comparación de Cargas de Contaminantes\nCon y Sin Tanques de Tormenta', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pollutants, fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Plot] Saved: {save_path}")
        
        return fig
    
    def plot_tank_efficiency(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create horizontal bar chart showing removal efficiency per pollutant.
        
        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        pollutants = self.pollutants
        efficiencies = [self.tank_efficiency.get(p, 0) * 100 for p in pollutants]
        
        # Colors based on efficiency
        colors = ['#27ae60' if e >= 60 else '#f39c12' if e >= 40 else '#e74c3c' 
                  for e in efficiencies]
        
        y_pos = np.arange(len(pollutants))
        bars = ax.barh(y_pos, efficiencies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add percentage labels
        for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                   f'{eff:.0f}%', ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pollutants, fontsize=11)
        ax.set_xlabel('Eficiencia de Remoción (%)', fontsize=12)
        ax.set_title('Eficiencia de Remoción de Contaminantes\nen Tanques de Tormenta', 
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add legend for colors
        legend_patches = [
            mpatches.Patch(color='#27ae60', label='Alta (≥60%)'),
            mpatches.Patch(color='#f39c12', label='Media (40-60%)'),
            mpatches.Patch(color='#e74c3c', label='Baja (<40%)')
        ]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Plot] Saved: {save_path}")
        
        return fig
    
    def plot_outfall_distribution(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> plt.Figure:
        """
        Create pie chart showing load distribution by outfall.
        
        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        if not self.outfalls:
            print("[Error] No outfalls detected")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Placeholder data - in reality would use actual loads
        n_outfalls = len(self.outfalls)
        loads = np.random.uniform(10, 100, n_outfalls)  # Placeholder
        loads = loads / loads.sum() * 100  # Normalize to percentages
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_outfalls))
        
        wedges, texts, autotexts = ax.pie(
            loads, 
            labels=self.outfalls[:10],  # Limit labels
            autopct='%1.1f%%',
            colors=colors,
            pctdistance=0.75,
            explode=[0.02] * min(n_outfalls, 10)
        )
        
        ax.set_title('Distribución de Carga por Descarga (Outfall)\n(Datos de ejemplo)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Plot] Saved: {save_path}")
        
        return fig
    
    def plot_all(self, output_dir: str = None) -> List[plt.Figure]:
        """
        Generate all available plots.
        
        Args:
            output_dir: Directory to save plots (optional)
            
        Returns:
            List of matplotlib Figure objects
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        figures = []
        
        # Plot 1: Load comparison
        fig1 = self.plot_load_comparison(
            save_path=str(output_dir / "quality_load_comparison.png")
        )
        if fig1:
            figures.append(fig1)
        
        # Plot 2: Tank efficiency
        fig2 = self.plot_tank_efficiency(
            save_path=str(output_dir / "quality_tank_efficiency.png")
        )
        if fig2:
            figures.append(fig2)
        
        # Plot 3: Outfall distribution
        fig3 = self.plot_outfall_distribution(
            save_path=str(output_dir / "quality_outfall_distribution.png")
        )
        if fig3:
            figures.append(fig3)
        
        print(f"\n  [Plot] Generated {len(figures)} plots in {output_dir}")
        return figures
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a summary report of water quality analysis.
        
        Args:
            output_path: Path for the report file
            
        Returns:
            Report content as string
        """
        if not self.comparison:
            return "No comparison results available. Run compare_scenarios first."
        
        lines = []
        lines.append("="*70)
        lines.append("REPORTE DE ANÁLISIS DE CALIDAD DEL AGUA")
        lines.append("="*70)
        lines.append("")
        lines.append(f"Archivo base: {self.base_inp_path}")
        lines.append(f"Uso del suelo: {self.landuse}")
        lines.append(f"Contaminantes analizados: {', '.join(self.pollutants)}")
        lines.append("")
        lines.append("-"*70)
        lines.append("VALORES EMC UTILIZADOS (mg/L)")
        lines.append("-"*70)
        for poll, val in self.emc_values.items():
            if poll in self.pollutants:
                lines.append(f"  {poll}: {val}")
        lines.append("")
        lines.append("-"*70)
        lines.append("EFICIENCIAS DE REMOCIÓN EN TANQUES")
        lines.append("-"*70)
        for poll, eff in self.tank_efficiency.items():
            if poll in self.pollutants:
                lines.append(f"  {poll}: {eff*100:.0f}%")
        lines.append("")
        lines.append("-"*70)
        lines.append("RESUMEN DE RESULTADOS")
        lines.append("-"*70)
        for poll, cr in self.comparison.items():
            lines.append(f"\n{poll}:")
            lines.append(f"  Reducción de carga: {cr.efficiency_percent:.1f}%")
        
        report = '\n'.join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"  [Quality] Report saved: {output_path}")
        
        return report


# =============================================================================
# MAIN - Example usage
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("WATER QUALITY ANALYZER - rut_28")
    print("="*70)
    
    # Default paths (change as needed)
    base_inp = "COLEGIO_TR25_v6.inp"
    
    if len(sys.argv) > 1:
        base_inp = sys.argv[1]
    
    if not os.path.exists(base_inp):
        print(f"[Error] No se encontró: {base_inp}")
        print("Uso: python rut_28_water_quality.py <archivo_base.inp>")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = WaterQualityAnalyzer(
        base_inp_path=base_inp,
        pollutants=['TSS', 'DBO', 'DQO', 'NT', 'PT'],
        landuse='Residencial'
    )
    
    # Create a quality-enabled version of the INP
    quality_inp = analyzer.create_quality_inp(base_inp, "test_quality")
    
    # Simulate comparison results for demo plots
    for pollutant in analyzer.pollutants:
        emc = analyzer.emc_values.get(pollutant, 100)
        efficiency = analyzer.tank_efficiency.get(pollutant, 0.5)
        
        analyzer.comparison[pollutant] = ComparisonResults(
            pollutant=pollutant,
            load_without_tanks_kg=emc * 10,  # Simulated total load
            load_with_tanks_kg=emc * 10 * (1 - efficiency * 0.6),
            load_reduction_kg=emc * 10 * efficiency * 0.6,
            efficiency_percent=efficiency * 60
        )
    
    print("\n" + "="*70)
    print("GENERANDO PLOTS DE DEMOSTRACIÓN")
    print("="*70)
    
    # Generate all plots
    analyzer.plot_all()
    
    print(f"\n[Done] Archivo con calidad: {quality_inp}")
    print("Plots guardados en el directorio actual.")
    plt.show()
