"""
WATER QUALITY ANALYZER (STRICT) - rut_28
========================================

STRICT = sin fallbacks:
- Si falta algo, explota con error y te dice qué.

Qué hace:
1) Genera INPs con calidad ([POLLUTANTS]/[BUILDUP]/[WASHOFF]/[TREATMENT]...)
2) Corre SWMM
3) Extrae series desde .out usando out.node_series(node, attr_index) (rápido)
4) Compara series (conc, carga incremental, carga acumulada) para TODOS outfalls y contaminantes
5) Grafica y devuelve estadísticas

Importante:
- Para "first flush" NO uses WASHOFF = EMC. Usa EXP o RATING.
- Este módulo usa WASHOFF EXP por defecto para que haya first flush.

Requisitos:
- pyswmm
- aenum
- numpy, pandas, matplotlib
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aenum import Enum
from pyswmm import Simulation, Output

import config
config.setup_sys_path()


# ============================================================
# Parámetros “normales” de arranque (ajústalos tú)
# ============================================================

# BUILDUP EXP: Coeff1=Bmax, Coeff2=tasa (1/día). Unidades internas de SWMM (masa/área).
DEFAULT_BUILDUP = {
    "TSS": (400.0, 0.50),
    "DBO": (60.0,  0.50),
}

# WASHOFF EXP: parámetros de arranque para ver first flush
DEFAULT_WASHOFF_EXP = {
    "TSS": (0.15, 1.20),
    "DBO": (0.10, 1.10),
}

# Tratamiento tipo settling por paso:
# C = C* + (P - C*) * EXP( -(k/DEPTH) * (DT/3600) )
# k en m/hr si DEPTH en m (SWMM SI).
DEFAULT_SETTLING = {
    "TSS": (0.50, 20.0),  # k, C*
    "DBO": (0.08, 5.0),
}

# Decay coeff (1/día) en [POLLUTANTS]. Si no quieres decay, 0.
DEFAULT_KDECAY = {
    "TSS": 0.0,
    "DBO": 0.0,
}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class SeriesPack:
    times: List[pd.Timestamp]
    flow_Lps: np.ndarray
    conc_mgL: np.ndarray
    load_kg_step: np.ndarray
    load_kg_cum: np.ndarray


@dataclass
class OutfallPollutantStats:
    outfall: str
    pollutant: str
    total_kg_base: float
    total_kg_sol: float
    red_kg: float
    red_pct: float
    peak_base_mgL: float
    peak_sol_mgL: float


@dataclass
class CompareStats:
    pollutant: str
    outfall_stats: List[OutfallPollutantStats] = field(default_factory=list)
    total_base_kg: float = 0.0
    total_sol_kg: float = 0.0
    total_red_kg: float = 0.0
    total_red_pct: float = 0.0


# ============================================================
# Helpers STRICT
# ============================================================

def _build_node_attribute_enum(n_pollutants: int):
    """
    Construye NodeAttribute dinámico según cantidad de contaminantes en el OUT.
    Base:
      INVERT_DEPTH=0, HYDRAULIC_HEAD=1, PONDED_VOLUME=2, LATERAL_INFLOW=3,
      TOTAL_INFLOW=4, FLOODING_LOSSES=5, POLLUT_CONC_0=6, ...
    """
    members = {
        "INVERT_DEPTH": 0,
        "HYDRAULIC_HEAD": 1,
        "PONDED_VOLUME": 2,
        "LATERAL_INFLOW": 3,
        "TOTAL_INFLOW": 4,
        "FLOODING_LOSSES": 5,
    }
    base = 6
    for i in range(n_pollutants):
        members[f"POLLUT_CONC_{i}"] = base + i
    return Enum("NodeAttribute", members)


def _normalize_ts_key(x: Any) -> pd.Timestamp:
    # SWMM/PySWMM normalmente usa datetime naive; normalizamos a pd.Timestamp naive
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    return t


def _series_to_np(series: Any, times: List[pd.Timestamp], label: str) -> np.ndarray:
    """
    Convierte lo que venga de out.node_series(...) a np.ndarray en el ORDEN de 'times'.

    En tu caso, node_series devuelve dict {datetime -> value}.
    También soporta pandas Series o secuencias ya ordenadas.

    STRICT:
    - Si falta un timestamp, error.
    - Si el tamaño no calza, error.
    """
    n = len(times)
    times_norm = [_normalize_ts_key(t) for t in times]

    # Caso 1: dict (lo tuyo)
    if isinstance(series, dict):
        d = {_normalize_ts_key(k): float(v) for k, v in series.items()}
        out = np.empty(n, dtype=float)
        for i, t in enumerate(times_norm):
            if t not in d:
                raise RuntimeError(f"STRICT: '{label}' no tiene valor para time={t}. (dict no alinea con out.times)")
            out[i] = d[t]
        return out

    # Caso 2: pandas Series con índice datetime
    if hasattr(series, "index") and hasattr(series, "values"):
        try:
            idx = [_normalize_ts_key(x) for x in list(series.index)]
            if len(idx) == n and idx[0] == times_norm[0] and idx[-1] == times_norm[-1]:
                arr = np.asarray(series.values, dtype=float)
                if arr.size != n:
                    raise RuntimeError(f"STRICT: '{label}' tamaño {arr.size} != {n}")
                return arr
            # si tiene index datetime pero no coincide, reindex STRICT
            s = pd.Series(series.values, index=idx, dtype=float)
            out = s.reindex(times_norm)
            if out.isna().any():
                bad = out[out.isna()].index[:5]
                raise RuntimeError(f"STRICT: '{label}' reindex deja NaN (ejemplos {list(bad)}). Ajusta REPORT_STEP/periodo.")
            return out.to_numpy(dtype=float)
        except Exception:
            # si falla, seguimos a los otros casos
            pass

    # Caso 3: secuencia/np array ya ordenada
    try:
        arr = np.asarray(series, dtype=float)
        if arr.size != n:
            raise RuntimeError(f"STRICT: '{label}' tamaño {arr.size} != {n}")
        return arr
    except Exception as e:
        raise RuntimeError(f"STRICT: No pude convertir '{label}' a np array. Tipo={type(series)}") from e


def _elapsed_hours(times: List[pd.Timestamp]) -> np.ndarray:
    t0 = times[0]
    return np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)


# ============================================================
# Clase principal
# ============================================================

class WaterQualityAnalyzerStrict:
    def __init__(
        self,
        base_inp_path: str,
        output_dir: Optional[str] = None,
        pollutants: Optional[List[str]] = None,
        landuse: str = "Residencial",
    ):
        self.base_inp_path = Path(base_inp_path)
        self.output_dir = Path(output_dir) if output_dir else (self.base_inp_path.parent / "water_quality_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pollutants = pollutants or ["TSS", "DBO"]
        self.landuse = landuse

        self.buildup_params = dict(DEFAULT_BUILDUP)
        self.washoff_exp_params = dict(DEFAULT_WASHOFF_EXP)
        self.settling_params = dict(DEFAULT_SETTLING)
        self.kdecay = dict(DEFAULT_KDECAY)

        self.outfalls: List[str] = []
        self.tanks: List[str] = []
        self.subcatchments: List[str] = []

    # --------------------------
    # Parse INP
    # --------------------------
    def _section_block(self, content: str, name: str) -> str:
        m = re.search(rf"\[{re.escape(name)}\](.*?)(?=\[|\Z)", content, re.DOTALL | re.IGNORECASE)
        return m.group(1) if m else ""

    def _parse_inp_elements(self, content: str) -> None:
        def ids_from_block(block: str) -> List[str]:
            ids = []
            for line in block.splitlines():
                s = line.strip()
                if not s or s.startswith(";"):
                    continue
                ids.append(s.split()[0])
            return ids

        self.outfalls = ids_from_block(self._section_block(content, "OUTFALLS"))
        self.tanks = ids_from_block(self._section_block(content, "STORAGE"))
        self.subcatchments = ids_from_block(self._section_block(content, "SUBCATCHMENTS"))

        if not self.outfalls:
            raise RuntimeError("STRICT: No encontré [OUTFALLS] o está vacío.")
        if not self.subcatchments:
            raise RuntimeError("STRICT: No encontré [SUBCATCHMENTS] o está vacío.")

    def _get_flow_units_from_inp(self, content: str) -> str:
        opt = self._section_block(content, "OPTIONS")
        for line in opt.splitlines():
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[0].upper() == "FLOW_UNITS":
                return parts[1].upper()
        raise RuntimeError("STRICT: No pude leer FLOW_UNITS desde [OPTIONS].")

    def _flow_to_Lps_factor(self, flow_units: str) -> float:
        u = flow_units.upper().strip()
        if u == "CMS":
            return 1000.0
        if u == "LPS":
            return 1.0
        if u == "CFS":
            return 28.316846592
        if u == "GPM":
            return 3.785411784 / 60.0
        if u == "MGD":
            return 3.785411784e6 / 86400.0
        if u == "MLD":
            return 1e6 / 86400.0
        raise RuntimeError(f"STRICT: FLOW_UNITS '{flow_units}' no soportado aquí.")

    # --------------------------
    # Calidad (generación)
    # --------------------------
    def _assert_no_quality_sections(self, content: str) -> None:
        for sec in ["POLLUTANTS", "LANDUSES", "COVERAGES", "BUILDUP", "WASHOFF", "TREATMENT"]:
            if re.search(rf"\[{sec}\]", content, re.IGNORECASE):
                raise RuntimeError(f"STRICT: El INP ya contiene [{sec}]. No duplico secciones.")

    def _treatment_expr_settling(self, pollutant: str) -> str:
        if pollutant not in self.settling_params:
            raise RuntimeError(f"STRICT: No hay settling_params para {pollutant}.")
        k_mph, cstar = self.settling_params[pollutant]
        return (
            f"C = {cstar:.3f} + ({pollutant} - {cstar:.3f})"
            f" * EXP(-({k_mph:.6f} / (DEPTH + 0.001)) * (DT / 3600))"
        )

    def _generate_quality_sections(self) -> str:
        lines = []

        lines.append("[POLLUTANTS]")
        lines.append(";;Name           Units  Crain  Cgw    Crdii  Kdecay SnowOnly Co-Pollutant  Co-Frac  Cdwf   Cinit")
        for p in self.pollutants:
            if p not in self.kdecay:
                raise RuntimeError(f"STRICT: falta kdecay para {p}")
            lines.append(f"{p:<16} MG/L   0      0      0      {self.kdecay[p]:.6g}  NO       *             0        0      0")
        lines.append("")

        lines.append("[LANDUSES]")
        lines.append(";;Name")
        lines.append(self.landuse)
        lines.append("")

        lines.append("[COVERAGES]")
        lines.append(";;Subcatchment   Land Use         Percent")
        for sc in self.subcatchments:
            lines.append(f"{sc:<16} {self.landuse:<16} 100")
        lines.append("")

        lines.append("[BUILDUP]")
        lines.append(";;Land Use       Pollutant        Function   Coeff1     Coeff2     Coeff3     Per Unit")
        for p in self.pollutants:
            if p not in self.buildup_params:
                raise RuntimeError(f"STRICT: falta buildup_params para {p}")
            bmax, rate = self.buildup_params[p]
            lines.append(f"{self.landuse:<16} {p:<16} EXP        {bmax:<10.3f} {rate:<10.3f} 0          AREA")
        lines.append("")

        lines.append("[WASHOFF]")
        lines.append(";;Land Use       Pollutant        Function   Coeff1     Coeff2     SweepRmvl  BmpRmvl")
        for p in self.pollutants:
            if p not in self.washoff_exp_params:
                raise RuntimeError(f"STRICT: falta washoff_exp_params para {p}")
            c1, c2 = self.washoff_exp_params[p]
            lines.append(f"{self.landuse:<16} {p:<16} EXP        {c1:<10.6f} {c2:<10.3f} 0          0")
        lines.append("")

        if self.tanks:
            lines.append("[TREATMENT]")
            lines.append(";;Node           Pollutant        Result = Expression")
            lines.append(";;Treatment Model: SETTLING (STRICT)")
            for tank in self.tanks:
                for p in self.pollutants:
                    expr = self._treatment_expr_settling(p)
                    lines.append(f"{tank:<16} {p:<16} {expr}")
            lines.append("")

        return "\n".join(lines)

    def create_quality_inp(self, source_inp: str, output_name: str, inject_quality: bool = True) -> str:
        src = Path(source_inp)
        if not src.exists():
            raise RuntimeError(f"STRICT: INP no existe: {src}")

        out_inp = self.output_dir / f"{output_name}_quality.inp"
        content = src.read_text(encoding="utf-8", errors="ignore")

        self._parse_inp_elements(content)

        if inject_quality:
            self._assert_no_quality_sections(content)
            quality = self._generate_quality_sections()

            if re.search(r"\[REPORT\]", content, re.IGNORECASE):
                content = re.sub(r"\[REPORT\]", quality + "\n[REPORT]", content, flags=re.IGNORECASE)
            else:
                content = content.rstrip() + "\n\n" + quality + "\n"

        out_inp.write_text(content, encoding="utf-8")
        return str(out_inp)

    # --------------------------
    # Run SWMM
    # --------------------------
    def run_swmm(self, inp_path: str) -> Tuple[str, str]:
        inp = Path(inp_path)
        if not inp.exists():
            raise RuntimeError(f"STRICT: INP no existe: {inp}")

        out_path = str(inp.with_suffix(".out"))
        rpt_path = str(inp.with_suffix(".rpt"))

        with Simulation(str(inp)) as sim:
            for _ in sim:
                pass

        if not Path(out_path).exists():
            raise RuntimeError(f"STRICT: No se generó .out: {out_path}")
        if not Path(rpt_path).exists():
            raise RuntimeError(f"STRICT: No se generó .rpt: {rpt_path}")

        return out_path, rpt_path

    # --------------------------
    # Output checks
    # --------------------------
    def _read_output_timebase(self, out: Output) -> List[pd.Timestamp]:
        times = [pd.Timestamp(t) for t in out.times]
        if len(times) < 2:
            raise RuntimeError("STRICT: output tiene <2 timesteps.")
        return times

    def _assert_same_period_and_step(self, t0: List[pd.Timestamp], t1: List[pd.Timestamp]) -> None:
        if t0[0] != t1[0] or t0[-1] != t1[-1]:
            raise RuntimeError(
                "STRICT: Los escenarios no tienen el mismo periodo de simulación.\n"
                f"  baseline: {t0[0]}  -> {t0[-1]}\n"
                f"  solution: {t1[0]}  -> {t1[-1]}\n"
                "Arregla START/END DATE/TIME y REPORT_START_DATE/TIME y REPORT_STEP para que sean iguales."
            )

        dt0 = np.diff(np.array([x.value for x in t0], dtype=np.int64))
        dt1 = np.diff(np.array([x.value for x in t1], dtype=np.int64))
        if dt0.size != dt1.size:
            raise RuntimeError(f"STRICT: Longitud de dt distinta: {dt0.size} vs {dt1.size}")
        if not np.array_equal(dt0, dt1):
            raise RuntimeError("STRICT: REPORT_STEP distinto o timebase distinto entre escenarios.")

    # --------------------------
    # Extracción de series por outfall/pollutant
    # --------------------------
    def _extract_flow_Lps(
        self,
        out: Output,
        outfall: str,
        times: List[pd.Timestamp],
        flow_attr_value: int,
        q_factor: float,
    ) -> np.ndarray:
        q_series = out.node_series(outfall, flow_attr_value)
        q_model = _series_to_np(q_series, times, f"{outfall}:flow")
        q_Lps = q_model * q_factor
        if np.any(q_Lps < 0):
            raise RuntimeError(f"STRICT: Flujo negativo en '{outfall}'. Define otra métrica si aplica.")
        return q_Lps

    def _extract_conc_mgL(
        self,
        out: Output,
        NA,
        outfall: str,
        times: List[pd.Timestamp],
        poll_idx: int,
    ) -> np.ndarray:
        conc_attr_name = f"POLLUT_CONC_{poll_idx}"
        if not hasattr(NA, conc_attr_name):
            raise RuntimeError(f"STRICT: NodeAttribute no tiene {conc_attr_name}. (enum sizing bug)")
        conc_attr_value = getattr(NA, conc_attr_name).value
        c_series = out.node_series(outfall, conc_attr_value)
        c = _series_to_np(c_series, times, f"{outfall}:conc_{poll_idx}")
        return c

    def _compute_loads(
        self,
        times: List[pd.Timestamp],
        q_Lps: np.ndarray,
        c_mgL: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(times)
        tvals = np.array([pd.Timestamp(t).value for t in times], dtype=np.int64)  # ns
        dt_s = np.diff(tvals) / 1e9  # segundos

        load_step = np.zeros(n, dtype=float)
        c_avg = 0.5 * (c_mgL[1:] + c_mgL[:-1])
        q_avg = 0.5 * (q_Lps[1:] + q_Lps[:-1])
        load_step[1:] = (c_avg * q_avg * dt_s) / 1_000_000.0
        load_cum = np.cumsum(load_step)
        return load_step, load_cum

    # --------------------------
    # Comparación completa
    # --------------------------
    def compare_timeseries(
        self,
        baseline_inp: str,
        solution_inp: str,
        scenario_names: Tuple[str, str] = ("Sin tanques", "Con tanques"),
        inject_quality: bool = True,
        run_simulations: bool = True,
        flow_metric: str = "TOTAL_INFLOW",
        make_plots: bool = True,
        plot_elapsed_hours: bool = True,
    ) -> Dict[str, CompareStats]:
        base_name, sol_name = scenario_names

        q0_inp = self.create_quality_inp(baseline_inp, "baseline", inject_quality=inject_quality)
        q1_inp = self.create_quality_inp(solution_inp, "solution", inject_quality=inject_quality)

        if run_simulations:
            out0_path, _ = self.run_swmm(q0_inp)
            out1_path, _ = self.run_swmm(q1_inp)
        else:
            out0_path = str(Path(q0_inp).with_suffix(".out"))
            out1_path = str(Path(q1_inp).with_suffix(".out"))
            if not Path(out0_path).exists() or not Path(out1_path).exists():
                raise RuntimeError("STRICT: run_simulations=False pero faltan .out.")

        # FLOW_UNITS desde INP baseline
        c0 = Path(q0_inp).read_text(encoding="utf-8", errors="ignore")
        flow_units = self._get_flow_units_from_inp(c0)
        q_factor = self._flow_to_Lps_factor(flow_units)

        stats_by_poll: Dict[str, CompareStats] = {p: CompareStats(pollutant=p) for p in self.pollutants}

        with Output(out0_path) as out0, Output(out1_path) as out1:
            t0 = self._read_output_timebase(out0)
            t1 = self._read_output_timebase(out1)
            self._assert_same_period_and_step(t0, t1)

            pol0 = list(out0.pollutants.keys()) if out0.pollutants else []
            pol1 = list(out1.pollutants.keys()) if out1.pollutants else []
            if pol0 != pol1:
                raise RuntimeError(f"STRICT: Lista de contaminantes distinta en outputs.\n  base={pol0}\n  sol ={pol1}")

            for p in self.pollutants:
                if p not in pol0:
                    raise RuntimeError(f"STRICT: El contaminante '{p}' no existe en el .out. Disponibles: {pol0}")

            poll_map = {p: pol0.index(p) for p in pol0}
            NA = _build_node_attribute_enum(len(pol0))

            if not hasattr(NA, flow_metric):
                raise RuntimeError(f"STRICT: flow_metric '{flow_metric}' no existe en NodeAttribute.")
            flow_attr_value = getattr(NA, flow_metric).value

            # outfalls STRICT: deben existir en outputs
            outfall_nodes = []
            for of in self.outfalls:
                if of not in out0.nodes:
                    raise RuntimeError(f"STRICT: Outfall '{of}' no está en output baseline.")
                if of not in out1.nodes:
                    raise RuntimeError(f"STRICT: Outfall '{of}' no está en output solution.")
                outfall_nodes.append(of)

            # Loop por outfall: extraer flujo una vez por escenario
            for outfall in outfall_nodes:
                q0_Lps = self._extract_flow_Lps(out0, outfall, t0, flow_attr_value, q_factor)
                q1_Lps = self._extract_flow_Lps(out1, outfall, t0, flow_attr_value, q_factor)

                for p in self.pollutants:
                    idx = int(poll_map[p])

                    c0_mgL = self._extract_conc_mgL(out0, NA, outfall, t0, idx)
                    c1_mgL = self._extract_conc_mgL(out1, NA, outfall, t0, idx)

                    load0_step, load0_cum = self._compute_loads(t0, q0_Lps, c0_mgL)
                    load1_step, load1_cum = self._compute_loads(t0, q1_Lps, c1_mgL)

                    total_base = float(load0_cum[-1])
                    total_sol = float(load1_cum[-1])
                    red_kg = total_base - total_sol
                    red_pct = (red_kg / total_base * 100.0) if total_base > 0 else 0.0

                    peak_base = float(np.max(c0_mgL))
                    peak_sol = float(np.max(c1_mgL))

                    stats_by_poll[p].outfall_stats.append(
                        OutfallPollutantStats(
                            outfall=outfall,
                            pollutant=p,
                            total_kg_base=total_base,
                            total_kg_sol=total_sol,
                            red_kg=red_kg,
                            red_pct=red_pct,
                            peak_base_mgL=peak_base,
                            peak_sol_mgL=peak_sol,
                        )
                    )

                    stats_by_poll[p].total_base_kg += total_base
                    stats_by_poll[p].total_sol_kg += total_sol

                    if make_plots:
                        base_pack = SeriesPack(t0, q0_Lps, c0_mgL, load0_step, load0_cum)
                        sol_pack = SeriesPack(t0, q1_Lps, c1_mgL, load1_step, load1_cum)
                        self._plot_compare_one(
                            pollutant=p,
                            outfall=outfall,
                            times=t0,
                            base_pack=base_pack,
                            sol_pack=sol_pack,
                            scenario_names=(base_name, sol_name),
                            plot_elapsed_hours=plot_elapsed_hours,
                        )

        for p in self.pollutants:
            s = stats_by_poll[p]
            s.total_red_kg = s.total_base_kg - s.total_sol_kg
            s.total_red_pct = (s.total_red_kg / s.total_base_kg * 100.0) if s.total_base_kg > 0 else 0.0

        print("WATER QUALITY ANALYZER (STRICT) - rut_28")
        print(f"FLOW_UNITS={flow_units}  factor_to_Lps={q_factor}")
        for p in self.pollutants:
            s = stats_by_poll[p]
            print("")
            print(f"{p}:")
            print(f"  WITHOUT tanks: {s.total_base_kg:.3f} kg")
            print(f"  WITH tanks:    {s.total_sol_kg:.3f} kg")
            print(f"  Reduction:     {s.total_red_kg:.3f} kg ({s.total_red_pct:.2f}%)")

        return stats_by_poll

    # --------------------------
    # Plot helper
    # --------------------------
    def _plot_compare_one(
        self,
        pollutant: str,
        outfall: str,
        times: List[pd.Timestamp],
        base_pack: SeriesPack,
        sol_pack: SeriesPack,
        scenario_names: Tuple[str, str],
        plot_elapsed_hours: bool,
    ) -> None:
        base_name, sol_name = scenario_names

        if plot_elapsed_hours:
            x = _elapsed_hours(times)
            xlabel = "Elapsed Time (hours)"
        else:
            x = pd.to_datetime(times)
            xlabel = "Tiempo"

        # Concentración
        fig1 = plt.figure(figsize=(12, 5))
        plt.plot(x, base_pack.conc_mgL, label=base_name)
        plt.plot(x, sol_pack.conc_mgL, label=sol_name)
        plt.title(f"{pollutant} - {outfall} - Concentración (mg/L)")
        plt.ylabel("mg/L")
        plt.xlabel(xlabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig1.tight_layout()
        p1 = self.output_dir / f"{pollutant}_{outfall}_conc_compare.png"
        fig1.savefig(p1, dpi=150)
        plt.close(fig1)

        # Carga acumulada
        fig2 = plt.figure(figsize=(12, 5))
        plt.plot(x, base_pack.load_kg_cum, label=f"{base_name} (kg)")
        plt.plot(x, sol_pack.load_kg_cum, label=f"{sol_name} (kg)")
        plt.title(f"{pollutant} - {outfall} - Carga acumulada (kg)")
        plt.ylabel("kg")
        plt.xlabel(xlabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig2.tight_layout()
        p2 = self.output_dir / f"{pollutant}_{outfall}_load_cum_compare.png"
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)

        # Reducción acumulada
        red_cum = base_pack.load_kg_cum - sol_pack.load_kg_cum
        fig3 = plt.figure(figsize=(12, 5))
        plt.plot(x, red_cum)
        plt.title(f"{pollutant} - {outfall} - Reducción acumulada (kg) ({base_name} - {sol_name})")
        plt.ylabel("kg")
        plt.xlabel(xlabel)
        plt.grid(True, alpha=0.3)
        fig3.tight_layout()
        p3 = self.output_dir / f"{pollutant}_{outfall}_reduction_cum.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)


# ============================================================
# MAIN ejemplo
# ============================================================
if __name__ == "__main__":
    from pathlib import Path
    import re

    print("WATER QUALITY ANALYZER (STRICT) - rut_28 | FIRST FLUSH (BUILDUP+WASHOFF)")

    baseline = Path(config.SWMM_FILE)
    solution = Path(r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\optimization_results\Seq_Iter_10\model_Seq_Iter_10.inp")

    if not baseline.exists():
        raise RuntimeError(f"STRICT: baseline INP no existe: {baseline}")
    if not solution.exists():
        raise RuntimeError(f"STRICT: solution INP no existe: {solution}")

    # ------------------------------------------------------------
    # Helper STRICT: setea/inyecta una opción dentro de [OPTIONS]
    # ------------------------------------------------------------
    def set_inp_option(src_inp: Path, dst_inp: Path, key: str, value: str) -> Path:
        txt = src_inp.read_text(encoding="utf-8", errors="ignore")

        m = re.search(r"\[OPTIONS\](.*?)(?=\[|\Z)", txt, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            raise RuntimeError("STRICT: INP no tiene sección [OPTIONS].")

        block = m.group(1)

        # quita líneas viejas de esa key
        new_lines = []
        found = False
        for line in block.splitlines():
            s = line.strip()
            if not s or s.startswith(";"):
                new_lines.append(line)
                continue

            parts = s.split()
            if len(parts) >= 2 and parts[0].upper() == key.upper():
                new_lines.append(f"{key} {value}")
                found = True
            else:
                new_lines.append(line)

        if not found:
            # inserta al final del bloque OPTIONS
            new_lines.append(f"{key} {value}")

        new_block = "\n".join(new_lines)
        txt2 = txt[: m.start(1)] + new_block + txt[m.end(1) :]

        dst_inp.write_text(txt2, encoding="utf-8")
        return dst_inp

    # ------------------------------------------------------------
    # 1) Copias "first flush" con DRY_DAYS>0 (carga inicial)
    # ------------------------------------------------------------
    outdir = baseline.parent / "water_quality_results"
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_ff = outdir / "baseline_firstflush.inp"
    solution_ff = outdir / "solution_firstflush.inp"

    # 7 días secos antecedentes es un arranque típico urbano
    baseline_ff = set_inp_option(baseline, baseline_ff, "DRY_DAYS", "7")
    solution_ff = set_inp_option(solution, solution_ff, "DRY_DAYS", "7")

    # ------------------------------------------------------------
    # 2) Analyzer + parámetros BUILDUP/WASHOFF "normales" (EXP)
    # ------------------------------------------------------------
    analyzer = WaterQualityAnalyzerStrict(
        base_inp_path=str(baseline_ff),
        pollutants=["TSS"],
        landuse="Residencial",
    )

    analyzer.buildup_params["TSS"] = (150.0, 0.4)  # Sube Bmax: más masa inicial (~8 millones kg posible, cargas >20 kg)
    analyzer.washoff_exp_params["TSS"] = (10.0, 0.4)  # C1 alto: más lavado (aumenta kg lavada 5-10x), C2 bajo para flush inicial
    analyzer.settling_params["TSS"] = (5.0, 10.0)  # k alto: más settling (reducción >10-20% si HRT >0.5h)
    
    # Agrega esto antes de stats = para debug en consola:
    print("Params TSS: buildup=", analyzer.buildup_params["TSS"], "washoff=", analyzer.washoff_exp_params["TSS"], "settling=", analyzer.settling_params["TSS"])

    # ------------------------------------------------------------
    # 3) Corre comparación y grafica
    # ------------------------------------------------------------
    stats = analyzer.compare_timeseries(
        baseline_inp=str(baseline_ff),
        solution_inp=str(solution_ff),
        scenario_names=("Sin tanques", "Con tanques"),
        inject_quality=True,
        run_simulations=True,
        flow_metric="TOTAL_INFLOW",
        make_plots=True,
        plot_elapsed_hours=True,
    )

    print(f"Plots guardados en: {analyzer.output_dir}")
    print("Done.")
