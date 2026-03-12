"""
Cross-TR Validator for Stormwater Tank Solutions
=================================================

Valida una solución diseñada para un TR específico comparándola
contra los baselines de múltiples TR.

Ejecuta 4 tipos de comparación:
1. Paso 1: Pre-proceso baselines (TR1, TR2, TR5, TR10, TR25, TR50)
2. Paso 2: Simular solución con todas las lluvias
3. Paso 3: Comparación funcional (sol_trX vs base_trX)
4. Paso 4: Comparación estadística cruzada (sol_tr25 vs base_trX)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import pickle
import json
from datetime import datetime

import config
config.setup_sys_path()

from rut_27_model_metrics import MetricExtractor, SystemMetrics
from rut_22_scenario_generator import generate_alternating_block_hyetograph


@dataclass
class TRMetrics:
    """Métricas extraídas para un modelo con un TR específico."""
    tr: int
    model_type: str  # 'baseline' o 'solution'
    inp_path: Path
    
    # Métricas del sistema
    flood_volume: float = 0.0
    flood_peak_flow: float = 0.0
    outfall_volume: float = 0.0
    outfall_peak_flow: float = 0.0
    network_health: float = 0.0
    network_utilization_mean: float = 0.0
    surcharged_links_count: int = 0
    flooded_nodes_count: int = 0
    
    # Hidrogramas (opcional, para gráficos detallados)
    flood_hydrograph: Dict = None
    outfall_hydrograph: Dict = None


@dataclass
class CrossComparison:
    """Resultado de comparación cruzada."""
    comparison_type: str  # 'functional' o 'statistical'
    tr_base: int  # TR del baseline
    tr_sol: int   # TR de la solución (simulación)
    
    # Métricas base
    base_flood_volume: float
    base_flood_peak: float
    base_outfall_volume: float
    base_outfall_peak: float
    base_network_health: float
    
    # Métricas solución
    sol_flood_volume: float
    sol_flood_peak: float
    sol_outfall_volume: float
    sol_outfall_peak: float
    sol_network_health: float
    
    # Diferencias
    diff_volume_abs: float
    diff_volume_pct: float
    diff_peak_abs: float
    diff_peak_pct: float
    diff_outfall_abs: float
    diff_outfall_pct: float
    diff_health_abs: float
    diff_health_pct: float
    
    # Ratios
    ratio_volume: float
    ratio_peak: float
    ratio_outfall: float
    ratio_health: float


class CrossTRValidator:
    """
    Valida una solución comparándola contra múltiples baselines de TR.
    
    Usage:
        validator = CrossTRValidator(
            baseline_inp=config.SWMM_FILE,
            solution_inp=solution_path,
            solution_design_tr=25,
            work_dir=output_dir
        )
        
        # Ejecutar todo el proceso
        results = validator.run_full_validation(tr_list=[1, 2, 5, 10, 25, 50])
    """
    
    def __init__(self, 
                 baseline_inp: Path,
                 solution_inp: Path,
                 solution_design_tr: int = 25,
                 work_dir: Path = None,
                 enable_caching: bool = True):
        """
        Args:
            baseline_inp: Ruta al modelo base sin tanques
            solution_inp: Ruta al modelo con tanques (solución)
            solution_design_tr: TR para el que fue diseñada la solución
            work_dir: Directorio de salida
            enable_caching: Si True, guarda métricas en cache para reutilizar
        """
        self.baseline_inp = Path(baseline_inp)
        self.solution_inp = Path(solution_inp)
        self.solution_tr = solution_design_tr
        self.enable_caching = enable_caching
        
        if work_dir is None:
            self.work_dir = Path(config.CODIGOS_DIR) / "optimization_results" / "cross_tr_validation"
        else:
            self.work_dir = Path(work_dir)
        
        self.cache_dir = self.work_dir / "cache"
        self.scenarios_dir = self.work_dir / "scenarios"
        self.figures_dir = self.work_dir / "figures"
        
        for d in [self.work_dir, self.cache_dir, self.scenarios_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Almacenamiento de métricas
        self.baseline_metrics: Dict[int, TRMetrics] = {}
        self.solution_metrics: Dict[int, TRMetrics] = {}
        self.comparisons: List[CrossComparison] = []
        
        print(f"[CrossTR] Inicializado:")
        print(f"  Baseline: {self.baseline_inp}")
        print(f"  Solution: {self.solution_inp}")
        print(f"  Design TR: {self.solution_tr}")
        print(f"  Work dir: {self.work_dir}")
    
    def run_full_validation(self, tr_list: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta el proceso completo de validación.
        
        Returns:
            (df_functional, df_statistical): Dos DataFrames con comparaciones
        """
        if tr_list is None:
            tr_list = [1, 2, 5, 10, 25, 50]
        
        print(f"\n{'='*80}")
        print(f"CROSS-TR VALIDATION - Full Process")
        print(f"{'='*80}")
        print(f"TR list: {tr_list}")
        print(f"{'='*80}\n")
        
        # PASO 1: Pre-proceso baselines
        print("\n[STEP 1/4] Pre-procesando baselines...")
        for tr in tr_list:
            self.baseline_metrics[tr] = self._extract_metrics(
                inp_path=self.baseline_inp,
                tr=tr,
                model_type='baseline'
            )
        
        # PASO 2: Simular solución con todas las lluvias
        print("\n[STEP 2/4] Simulando solución con múltiples TR...")
        for tr in tr_list:
            self.solution_metrics[tr] = self._extract_metrics(
                inp_path=self.solution_inp,
                tr=tr,
                model_type='solution'
            )
        
        # PASO 3: Comparación funcional (sol_trX vs base_trX)
        print("\n[STEP 3/4] Comparación funcional (mismo TR)...")
        functional_comparisons = []
        for tr in tr_list:
            comp = self._compare_pair(
                base_metrics=self.baseline_metrics[tr],
                sol_metrics=self.solution_metrics[tr],
                tr_base=tr,
                tr_sol=tr,
                comparison_type='functional'
            )
            functional_comparisons.append(comp)
        
        # PASO 4: Comparación estadística cruzada (sol_tr25 vs base_trX)
        print("\n[STEP 4/4] Comparación estadística cruzada...")
        statistical_comparisons = []
        for tr in tr_list:
            if tr != self.solution_tr:  # Excluir TR de diseño para evitar redundancia
                comp = self._compare_pair(
                    base_metrics=self.baseline_metrics[tr],
                    sol_metrics=self.solution_metrics[self.solution_tr],
                    tr_base=tr,
                    tr_sol=self.solution_tr,
                    comparison_type='statistical'
                )
                statistical_comparisons.append(comp)
        
        self.comparisons = functional_comparisons + statistical_comparisons
        
        # Convertir a DataFrames
        df_func = self._comparisons_to_df(functional_comparisons)
        df_stat = self._comparisons_to_df(statistical_comparisons)
        
        # Guardar resultados
        self._save_results(df_func, df_stat)
        
        return df_func, df_stat
    
    def _extract_metrics(self, inp_path: Path, tr: int, model_type: str) -> TRMetrics:
        """Extrae métricas para un modelo con un TR específico."""
        
        # Verificar cache
        cache_key = f"{model_type}_TR{tr:03d}_{inp_path.stem}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if self.enable_caching and cache_file.exists():
            print(f"  [Cache] Cargando {model_type} TR{tr}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"  [Extract] {model_type} TR{tr}...")
        
        # Asegurar que existe el escenario
        scenario_inp = self._ensure_scenario(inp_path, tr)
        
        # Extraer métricas con MetricExtractor
        extractor = MetricExtractor(
            swmm_file=str(scenario_inp),
            predios_path=config.PREDIOS_FILE
        )
        
        metrics = extractor.metrics
        
        # Crear objeto TRMetrics
        tr_metrics = TRMetrics(
            tr=tr,
            model_type=model_type,
            inp_path=scenario_inp,
            flood_volume=metrics.total_flooding_volume,
            flood_peak_flow=metrics.total_max_flooding_flow,
            outfall_volume=metrics.total_outfall_volume,
            outfall_peak_flow=metrics.total_max_outfall_flow,
            network_health=metrics.network_health_score,
            network_utilization_mean=metrics.system_mean_utilization,
            surcharged_links_count=metrics.surcharged_links_count,
            flooded_nodes_count=metrics.flooded_nodes_count
        )
        
        # Guardar en cache
        if self.enable_caching:
            with open(cache_file, 'wb') as f:
                pickle.dump(tr_metrics, f)
        
        return tr_metrics
    
    def _ensure_scenario(self, base_inp: Path, tr: int) -> Path:
        """Genera escenario con hietograma del TR especificado."""
        
        scenario_name = f"{base_inp.stem}_TR{tr:03d}.inp"
        scenario_path = self.scenarios_dir / scenario_name
        
        if scenario_path.exists():
            return scenario_path
        
        print(f"    [Gen] Creando escenario TR{tr}...")
        
        # Leer base
        with open(base_inp, 'r', encoding='latin-1', errors='replace') as f:
            base_content = f.read()
        
        # Generar hietograma
        hyeto_df = generate_alternating_block_hyetograph(
            tr_years=tr,
            duration_min=60,
            dt_min=5
        )
        
        # Generar INP modificado
        from rut_22_scenario_generator import generate_inp_file
        
        generate_inp_file(
            base_content=base_content,
            tr=tr,
            hyetograph_df=hyeto_df,
            output_path=scenario_path
        )
        
        return scenario_path
    
    def _compare_pair(self, base_metrics: TRMetrics, sol_metrics: TRMetrics,
                     tr_base: int, tr_sol: int, comparison_type: str) -> CrossComparison:
        """Compara un par de métricas base vs solución."""
        
        def safe_diff(sol, base):
            return sol - base
        
        def safe_pct(sol, base):
            return ((sol - base) / base * 100) if base > 0 else 0
        
        def safe_ratio(sol, base):
            return sol / base if base > 0 else 0
        
        return CrossComparison(
            comparison_type=comparison_type,
            tr_base=tr_base,
            tr_sol=tr_sol,
            
            base_flood_volume=base_metrics.flood_volume,
            base_flood_peak=base_metrics.flood_peak_flow,
            base_outfall_volume=base_metrics.outfall_volume,
            base_outfall_peak=base_metrics.outfall_peak_flow,
            base_network_health=base_metrics.network_health,
            
            sol_flood_volume=sol_metrics.flood_volume,
            sol_flood_peak=sol_metrics.flood_peak_flow,
            sol_outfall_volume=sol_metrics.outfall_volume,
            sol_outfall_peak=sol_metrics.outfall_peak_flow,
            sol_network_health=sol_metrics.network_health,
            
            diff_volume_abs=safe_diff(sol_metrics.flood_volume, base_metrics.flood_volume),
            diff_volume_pct=safe_pct(sol_metrics.flood_volume, base_metrics.flood_volume),
            diff_peak_abs=safe_diff(sol_metrics.flood_peak_flow, base_metrics.flood_peak_flow),
            diff_peak_pct=safe_pct(sol_metrics.flood_peak_flow, base_metrics.flood_peak_flow),
            diff_outfall_abs=safe_diff(sol_metrics.outfall_volume, base_metrics.outfall_volume),
            diff_outfall_pct=safe_pct(sol_metrics.outfall_volume, base_metrics.outfall_volume),
            diff_health_abs=safe_diff(sol_metrics.network_health, base_metrics.network_health),
            diff_health_pct=safe_pct(sol_metrics.network_health, base_metrics.network_health),
            
            ratio_volume=safe_ratio(sol_metrics.flood_volume, base_metrics.flood_volume),
            ratio_peak=safe_ratio(sol_metrics.flood_peak_flow, base_metrics.flood_peak_flow),
            ratio_outfall=safe_ratio(sol_metrics.outfall_volume, base_metrics.outfall_volume),
            ratio_health=safe_ratio(sol_metrics.network_health, base_metrics.network_health)
        )
    
    def _comparisons_to_df(self, comparisons: List[CrossComparison]) -> pd.DataFrame:
        """Convierte lista de comparaciones a DataFrame."""
        data = []
        for c in comparisons:
            data.append({
                'comparison_type': c.comparison_type,
                'tr_base': c.tr_base,
                'tr_sol': c.tr_sol,
                'base_flood_volume': c.base_flood_volume,
                'base_flood_peak': c.base_flood_peak,
                'base_outfall_volume': c.base_outfall_volume,
                'base_network_health': c.base_network_health,
                'sol_flood_volume': c.sol_flood_volume,
                'sol_flood_peak': c.sol_flood_peak,
                'sol_outfall_volume': c.sol_outfall_volume,
                'sol_network_health': c.sol_network_health,
                'diff_volume_pct': c.diff_volume_pct,
                'diff_peak_pct': c.diff_peak_pct,
                'diff_outfall_pct': c.diff_outfall_pct,
                'diff_health_pct': c.diff_health_pct,
                'ratio_volume': c.ratio_volume,
                'ratio_peak': c.ratio_peak,
                'ratio_outfall': c.ratio_outfall,
                'ratio_health': c.ratio_health
            })
        return pd.DataFrame(data)
    
    def _save_results(self, df_func: pd.DataFrame, df_stat: pd.DataFrame):
        """Guarda todos los resultados y genera figuras."""
        
        # Guardar CSVs
        df_func.to_csv(self.work_dir / "comparison_functional.csv", index=False)
        df_stat.to_csv(self.work_dir / "comparison_statistical.csv", index=False)
        
        # Generar figuras
        self._plot_functional_comparison(df_func)
        self._plot_statistical_comparison(df_stat)
        self._plot_ratios_comparison(df_func, df_stat)
        
        # Generar resumen
        self._generate_summary_text(df_func, df_stat)
        
        print(f"\n[CrossTR] Resultados guardados en: {self.work_dir}")
    
    def _plot_functional_comparison(self, df: pd.DataFrame):
        """Gráfico de comparación funcional."""
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        tr_labels = [f"TR{int(r)}" for r in df['tr_base']]
        x = np.arange(len(tr_labels))
        width = 0.35
        
        # Volumen inundación
        ax = axes[0, 0]
        ax.bar(x - width/2, df['base_flood_volume'], width, label='Baseline', alpha=0.8, color='coral')
        ax.bar(x + width/2, df['sol_flood_volume'], width, label='Solución', alpha=0.8, color='skyblue')
        ax.set_ylabel('Volumen (m³)')
        ax.set_title('Volumen de Inundación - Comparación Funcional')
        ax.set_xticks(x)
        ax.set_xticklabels(tr_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Caudal pico
        ax = axes[0, 1]
        ax.bar(x - width/2, df['base_flood_peak'], width, label='Baseline', alpha=0.8, color='coral')
        ax.bar(x + width/2, df['sol_flood_peak'], width, label='Solución', alpha=0.8, color='skyblue')
        ax.set_ylabel('Caudal (m³/s)')
        ax.set_title('Caudal Pico - Comparación Funcional')
        ax.set_xticks(x)
        ax.set_xticklabels(tr_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Volumen outfall
        ax = axes[1, 0]
        ax.bar(x - width/2, df['base_outfall_volume'], width, label='Baseline', alpha=0.8, color='coral')
        ax.bar(x + width/2, df['sol_outfall_volume'], width, label='Solución', alpha=0.8, color='skyblue')
        ax.set_ylabel('Volumen (m³)')
        ax.set_title('Outfall - Comparación Funcional')
        ax.set_xticks(x)
        ax.set_xticklabels(tr_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Network Health
        ax = axes[1, 1]
        ax.bar(x - width/2, df['base_network_health'], width, label='Baseline', alpha=0.8, color='coral')
        ax.bar(x + width/2, df['sol_network_health'], width, label='Solución', alpha=0.8, color='skyblue')
        ax.set_ylabel('Health (0-1)')
        ax.set_title('Network Health - Comparación Funcional')
        ax.set_xticks(x)
        ax.set_xticklabels(tr_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "01_functional_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_comparison(self, df: pd.DataFrame):
        """Gráfico de comparación estadística."""
        if df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        tr_labels = [f"TR{int(r)}" for r in df['tr_base']]
        x = np.arange(len(tr_labels))
        
        # Barras de volumen
        ax.bar(x - 0.2, df['base_flood_volume'], 0.4, label='Baseline', alpha=0.7, color='lightcoral')
        ax.bar(x + 0.2, df['sol_flood_volume'], 0.4, label=f'Solución TR{self.solution_tr}', alpha=0.7, color='steelblue')
        
        # Líneas de diferencia porcentual
        ax2 = ax.twinx()
        ax2.plot(x, df['diff_volume_pct'], 'go-', linewidth=2, markersize=8, label='% Diferencia')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Período de Retorno Comparado', fontsize=12)
        ax.set_ylabel('Volumen Inundado (m³)', fontsize=12)
        ax2.set_ylabel('Diferencia Porcentual (%)', fontsize=12, color='green')
        ax.set_title(f'Comparación Estadística: Solución TR{self.solution_tr} vs Baselines', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tr_labels)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "02_statistical_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_ratios_comparison(self, df_func: pd.DataFrame, df_stat: pd.DataFrame):
        """Gráfico de ratios."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Ratios funcionales
        if not df_func.empty:
            tr_labels = [f"TR{int(r)}" for r in df_func['tr_base']]
            x = np.arange(len(tr_labels))
            
            ax1.plot(x, df_func['ratio_volume'], 'o-', label='Volumen', linewidth=2, markersize=8)
            ax1.plot(x, df_func['ratio_peak'], 's-', label='Caudal Pico', linewidth=2, markersize=8)
            ax1.plot(x, df_func['ratio_outfall'], '^-', label='Outfall', linewidth=2, markersize=8)
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
            ax1.set_xlabel('TR')
            ax1.set_ylabel('Ratio (Solución / Baseline)')
            ax1.set_title('Ratios - Comparación Funcional')
            ax1.set_xticks(x)
            ax1.set_xticklabels(tr_labels)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Ratios estadísticos
        if not df_stat.empty:
            tr_labels = [f"TR{int(r)}" for r in df_stat['tr_base']]
            x = np.arange(len(tr_labels))
            
            ax2.plot(x, df_stat['ratio_volume'], 'o-', label='Volumen', linewidth=2, markersize=8, color='purple')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
            ax2.set_xlabel('TR Comparado')
            ax2.set_ylabel('Ratio (Solución / Baseline)')
            ax2.set_title(f'Ratios - Solución TR{self.solution_tr} vs Baselines')
            ax2.set_xticks(x)
            ax2.set_xticklabels(tr_labels)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "03_ratios.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_text(self, df_func: pd.DataFrame, df_stat: pd.DataFrame):
        """Genera archivo de texto con resumen."""
        summary_path = self.work_dir / "summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CROSS-TR VALIDATION SUMMARY\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Solución diseñada para TR{self.solution_tr}\n")
            f.write("="*80 + "\n\n")
            
            # Comparación funcional
            f.write("--- COMPARACIÓN FUNCIONAL (mismo TR vs mismo TR) ---\n\n")
            for _, row in df_func.iterrows():
                tr = int(row['tr_base'])
                f.write(f"TR{tr}:\n")
                f.write(f"  Volumen Base: {row['base_flood_volume']:.1f} m³\n")
                f.write(f"  Volumen Sol:  {row['sol_flood_volume']:.1f} m³\n")
                f.write(f"  Diferencia:   {row['diff_volume_pct']:.1f}%\n")
                f.write(f"  Ratio:        {row['ratio_volume']:.2f}\n\n")
            
            # Comparación estadística
            f.write("\n--- COMPARACIÓN ESTADÍSTICA (Sol TR{} vs Baselines) ---\n\n".format(self.solution_tr))
            for _, row in df_stat.iterrows():
                tr = int(row['tr_base'])
                f.write(f"vs Baseline TR{tr}:\n")
                f.write(f"  Volumen Base TR{tr}:  {row['base_flood_volume']:.1f} m³\n")
                f.write(f"  Volumen Sol TR{self.solution_tr}: {row['sol_flood_volume']:.1f} m³\n")
                f.write(f"  Diferencia:          {row['diff_volume_pct']:.1f}%\n")
                f.write(f"  Ratio:               {row['ratio_volume']:.2f}\n")
                
                if row['ratio_volume'] < 1.0:
                    f.write(f"  → La solución MEJORA vs evento TR{tr} sin intervención\n\n")
                else:
                    f.write(f"  → La solución tiene MAYOR inundación que TR{tr} base\n\n")
            
            # Conclusión
            f.write("="*80 + "\n")
            f.write("CONCLUSIONES:\n")
            if not df_func.empty:
                avg_reduction = df_func['diff_volume_pct'].mean()
                f.write(f"- Reducción promedio (comparación funcional): {avg_reduction:.1f}%\n")
            if not df_stat.empty:
                ratios = df_stat['ratio_volume'].values
                f.write(f"- Ratio promedio vs baselines: {ratios.mean():.2f}\n")
                f.write(f"- Mejor ratio: {ratios.min():.2f} (vs TR{df_stat.loc[df_stat['ratio_volume'].idxmin(), 'tr_base']:.0f})\n")
                f.write(f"- Peor ratio: {ratios.max():.2f} (vs TR{df_stat.loc[df_stat['ratio_volume'].idxmax(), 'tr_base']:.0f})\n")
            f.write("="*80 + "\n")
        
        print(f"[Guardado] Resumen: {summary_path}")


# Función de conveniencia para integración
def run_cross_validation(solution_inp: Path, 
                        solution_tr: int,
                        baseline_inp: Path = None,
                        work_dir: Path = None,
                        tr_list: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Función de conveniencia para ejecutar validación cruzada.
    
    Returns:
        (df_functional, df_statistical)
    """
    if baseline_inp is None:
        baseline_inp = config.SWMM_FILE
    
    validator = CrossTRValidator(
        baseline_inp=baseline_inp,
        solution_inp=solution_inp,
        solution_design_tr=solution_tr,
        work_dir=work_dir
    )
    
    return validator.run_full_validation(tr_list=tr_list)


if __name__ == "__main__":
    # Ejemplo de uso
    solution_path = Path(config.CODIGOS_DIR) / "optimization_results" / "Seq_Iter_16" / "model_Seq_Iter_16.inp"
    
    if solution_path.exists():
        df_func, df_stat = run_cross_validation(
            solution_inp=solution_path,
            solution_tr=25,
            tr_list=[1, 2, 5, 10, 25]
        )
        
        print("\n" + "="*80)
        print("COMPARACIÓN FUNCIONAL:")
        print(df_func[['tr_base', 'diff_volume_pct', 'ratio_volume']].to_string(index=False))
        print("\n" + "="*80)
        print("COMPARACIÓN ESTADÍSTICA:")
        print(df_stat[['tr_base', 'diff_volume_pct', 'ratio_volume']].to_string(index=False))
    else:
        print(f"No se encuentra: {solution_path}")
