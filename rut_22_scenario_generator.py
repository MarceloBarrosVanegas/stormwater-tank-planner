"""
rut_22_scenario_generator.py
============================
Genera archivos SWMM .inp para múltiples períodos de retorno (TR) usando la
ecuación IDF del DAC - Aeropuerto y el Método de Bloques Alternos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ECUACIÓN IDF — DAC Aeropuerto (Cuadro A4):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  I(t, T) = (13.9378 · ln(T) + 40.7176) / (35.5037 + t)^0.9997

  Donde:
    t  = duración [minutos]
    T  = período de retorno [años]
    I  = INTENSIDAD [mm/min]   ← unidades de salida

  IMPORTANTE SOBRE EL LOGARITMO:
    El documento fuente (Cuadro A4) escribe "log T", pero al verificar
    contra la tabla de precipitaciones máximas se comprueba que usa el
    LOGARITMO NATURAL (ln), no log base 10.

    Verificación TR=5, t=60 min:
      Con ln(5):   I = 0.6614 mm/min → P = 39.7 mm  (tabla: 40.1 mm) ✓
      Con log10(5): I = 0.5286 mm/min → P = 31.7 mm  (tabla: 40.1 mm) ✗

    RMSE contra tabla completa:  ln → 0.44 mm  |  log10 → 10.77 mm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLA DE REFERENCIA — Cuadro A4 (DAC Aeropuerto, Precipitación máxima en mm):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  t(min)  TR=3   TR=5   TR=10  TR=15  TR=25  TR=30  TR=50
     5     7.1    8.0    9.2    9.9   10.8   11.2   12.1
    10    12.6   14.2   16.4   17.7   19.3   19.8   21.4
    15    17.0   19.2   22.1   23.8   26.0   26.7   28.9
    20    20.6   23.2   26.7   28.8   31.4   32.4   35.0
    25    23.6   26.6   30.6   33.0   36.0   37.1   40.1
    30    26.1   29.4   33.9   36.5   39.9   41.0   44.4
    35    28.3   31.8   36.7   39.6   43.2   44.4   48.0
    40    30.1   33.9   39.1   42.2   46.0   47.4   51.2
    45    31.8   35.8   41.3   44.5   48.5   49.9   54.0
    50    33.2   37.4   43.1   46.5   50.7   52.2   56.4
    55    34.5   38.9   44.8   48.3   52.7   54.2   58.6
    60    35.6   40.1   46.3   49.9   54.4   56.0   60.6
   120    43.6   49.1   56.6   61.0   66.6   68.5   74.1
   240    49.1   55.3   63.8   68.7   74.9   77.2   83.4
   360    51.2   57.7   66.6   71.7   78.2   80.6   87.1
   540    52.8   59.5   68.6   73.9   80.6   83.0   89.7
   720    53.6   60.4   69.6   75.0   81.8   84.3   91.1
   960    54.2   61.1   70.4   75.9   82.8   85.2   92.1
  1200    54.6   61.5   70.9   76.4   83.4   85.8   92.8
  1440    54.8   61.8   71.3   76.8   83.8   86.2   93.2
  1800    55.1   62.1   71.6   77.2   84.2   86.7   93.7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIGURACIÓN DE GENERACIÓN:
  - Duración tormenta : 60 minutos
  - Paso de tiempo    : 5 minutos
  - Períodos de retorno: 1, 2, 5, 10, 25, 50, 100 años

USO:
  python rut_22_scenario_generator.py           # Genera todos los .inp
  python rut_22_scenario_generator.py --validate # Verifica ecuación vs tabla
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import config

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_INP_PATH = config.SWMM_FILE
OUTPUT_DIR    = config.CODIGOS_DIR / "scenarios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION_MIN   = 60
TIME_STEP_MIN  = 5
RETURN_PERIODS = [1, 2, 5, 10, 25, 50, 100]

# =============================================================================
# ECUACIÓN IDF
# =============================================================================

def calculate_intensity(duration_min: float, tr_years: float) -> float:
    """
    Calcula la intensidad de precipitación usando la ecuación IDF del
    DAC - Aeropuerto (Cuadro A4).

    ECUACIÓN:
        I(t, T) = (13.9378 · ln(T) + 40.7176) / (35.5037 + t)^0.9997

    El documento escribe "log T" pero el logaritmo es NATURAL (ln).
    Ver encabezado del módulo para la verificación numérica completa.

    Args:
        duration_min : Duración t en minutos.
        tr_years     : Período de retorno T en años.

    Returns:
        Intensidad en mm/min.
    """
    numerator   = 13.9378 * math.log(tr_years) + 40.7176   # ln, no log10
    denominator = (35.5037 + duration_min) ** 0.9997
    return numerator / denominator  # mm/min


def calculate_depth(duration_min: float, tr_years: float) -> float:
    """
    Calcula la precipitación acumulada máxima P(t, T) en mm.

    P = I(t, T) [mm/min] × t [min]  →  mm

    Equivale a los valores del Cuadro A4 (Cantidad de precipitación máxima).
    """
    return calculate_intensity(duration_min, tr_years) * duration_min


# =============================================================================
# VALIDACIÓN CONTRA CUADRO A4
# =============================================================================

# Tabla de referencia del Cuadro A4 — DAC Aeropuerto
# Formato: {t_min: {TR: depth_mm, ...}, ...}
CUADRO_A4 = {
    5:    {3: 7.1,  5: 8.0,  10: 9.2,  15: 9.9,  25: 10.8, 30: 11.2, 50: 12.1},
    10:   {3: 12.6, 5: 14.2, 10: 16.4, 15: 17.7, 25: 19.3, 30: 19.8, 50: 21.4},
    15:   {3: 17.0, 5: 19.2, 10: 22.1, 15: 23.8, 25: 26.0, 30: 26.7, 50: 28.9},
    20:   {3: 20.6, 5: 23.2, 10: 26.7, 15: 28.8, 25: 31.4, 30: 32.4, 50: 35.0},
    25:   {3: 23.6, 5: 26.6, 10: 30.6, 15: 33.0, 25: 36.0, 30: 37.1, 50: 40.1},
    30:   {3: 26.1, 5: 29.4, 10: 33.9, 15: 36.5, 25: 39.9, 30: 41.0, 50: 44.4},
    35:   {3: 28.3, 5: 31.8, 10: 36.7, 15: 39.6, 25: 43.2, 30: 44.4, 50: 48.0},
    40:   {3: 30.1, 5: 33.9, 10: 39.1, 15: 42.2, 25: 46.0, 30: 47.4, 50: 51.2},
    45:   {3: 31.8, 5: 35.8, 10: 41.3, 15: 44.5, 25: 48.5, 30: 49.9, 50: 54.0},
    50:   {3: 33.2, 5: 37.4, 10: 43.1, 15: 46.5, 25: 50.7, 30: 52.2, 50: 56.4},
    55:   {3: 34.5, 5: 38.9, 10: 44.8, 15: 48.3, 25: 52.7, 30: 54.2, 50: 58.6},
    60:   {3: 35.6, 5: 40.1, 10: 46.3, 15: 49.9, 25: 54.4, 30: 56.0, 50: 60.6},
    120:  {3: 43.6, 5: 49.1, 10: 56.6, 15: 61.0, 25: 66.6, 30: 68.5, 50: 74.1},
    240:  {3: 49.1, 5: 55.3, 10: 63.8, 15: 68.7, 25: 74.9, 30: 77.2, 50: 83.4},
    360:  {3: 51.2, 5: 57.7, 10: 66.6, 15: 71.7, 25: 78.2, 30: 80.6, 50: 87.1},
    540:  {3: 52.8, 5: 59.5, 10: 68.6, 15: 73.9, 25: 80.6, 30: 83.0, 50: 89.7},
    720:  {3: 53.6, 5: 60.4, 10: 69.6, 15: 75.0, 25: 81.8, 30: 84.3, 50: 91.1},
    960:  {3: 54.2, 5: 61.1, 10: 70.4, 15: 75.9, 25: 82.8, 30: 85.2, 50: 92.1},
    1200: {3: 54.6, 5: 61.5, 10: 70.9, 15: 76.4, 25: 83.4, 30: 85.8, 50: 92.8},
    1440: {3: 54.8, 5: 61.8, 10: 71.3, 15: 76.8, 25: 83.8, 30: 86.2, 50: 93.2},
    1800: {3: 55.1, 5: 62.1, 10: 71.6, 15: 77.2, 25: 84.2, 30: 86.7, 50: 93.7},
}


def validate_against_table():
    """
    Imprime una comparación entre los valores calculados y el Cuadro A4.
    Ejecutar con: python rut_22_scenario_generator.py --validate
    """
    print("\n" + "=" * 80)
    print("VALIDACIÓN IDF — Calculado vs. Cuadro A4 (DAC Aeropuerto)")
    print("Precipitación acumulada P(t,T) = I(t,T) × t  [mm]")
    print("=" * 80)
    header = f"{'t':>6} {'TR':>4} | {'Tabla':>7} | {'Calc':>7} | {'Error':>7} | {'Error%':>7}"
    print(header)
    print("-" * 60)

    errors_sq = []
    for t, trs in sorted(CUADRO_A4.items()):
        for TR, ref in sorted(trs.items()):
            calc = calculate_depth(t, TR)
            err  = calc - ref
            pct  = 100 * err / ref
            errors_sq.append(err ** 2)
            flag = "  ←" if abs(pct) > 2.0 else ""
            print(f"{t:>6} {TR:>4} | {ref:>7.1f} | {calc:>7.2f} | {err:>+7.3f} | {pct:>+6.2f}%{flag}")

    rmse = math.sqrt(sum(errors_sq) / len(errors_sq))
    print("-" * 60)
    print(f"RMSE total: {rmse:.4f} mm  (diferencias debidas al redondeo de la tabla)")
    print("=" * 80)
    print("\nResumen para la duración de diseño (60 min):")
    print(f"{'TR':>4} | {'Tabla':>7} | {'Calc':>7} | {'Error':>7}")
    print("-" * 35)
    for TR, ref in sorted(CUADRO_A4[60].items()):
        calc = calculate_depth(60, TR)
        print(f"{TR:>4} | {ref:>7.1f} | {calc:>7.2f} | {calc - ref:>+7.3f}")
    print()


# =============================================================================
# MÉTODO DE BLOQUES ALTERNOS
# =============================================================================

def generate_alternating_block_hyetograph(tr_years: float,
                                          duration_min: int,
                                          dt_min: int) -> pd.DataFrame:
    """
    Genera el hietograma de diseño usando el Método de Bloques Alternos.

    Pasos:
        1. Calcula la profundidad acumulada P(t) = I(t,T) × t para cada
           múltiplo de dt_min. Unidades: mm.
        2. Obtiene las profundidades incrementales (bloque por bloque).
        3. Ordena de mayor a menor y redistribuye en orden alterno con el
           máximo centrado.

    Args:
        tr_years    : Período de retorno [años].
        duration_min: Duración total de la tormenta [minutos].
        dt_min      : Paso de tiempo [minutos].

    Returns:
        DataFrame con columnas:
            Offset_Min      — tiempo de inicio del bloque [min]
            Time_Str        — formato "H:MM" para SWMM
            Block_Depth_mm  — profundidad del bloque [mm]
            Intensity_mm_h  — intensidad equivalente [mm/h]
    """
    n_blocks = int(duration_min / dt_min)

    # ── 1. Profundidades acumuladas P(t) en los tiempos t = dt, 2·dt, ... ──
    durations        = np.arange(dt_min, duration_min + dt_min, dt_min)  # [5,10,...,60] min
    cum_depths       = np.array([calculate_depth(float(d), tr_years) for d in durations])  # mm

    # ── 2. Profundidades incrementales por bloque ──
    incremental      = np.diff(np.concatenate([[0.0], cum_depths]))      # mm/bloque

    # ── 3. Ordenar descendente y redistribuir en orden alterno ──
    sorted_blocks    = np.sort(incremental)[::-1]
    hyetograph       = np.zeros(n_blocks)

    # El bloque máximo se ubica en la posición central (índice n_blocks//2)
    # luego se alternan izquierda/derecha, izquierda/derecha...
    right_idx = n_blocks // 2
    left_idx  = right_idx - 1

    for i, depth in enumerate(sorted_blocks):
        if i % 2 == 0:                  # pares → derecha
            if right_idx < n_blocks:
                hyetograph[right_idx] = depth
                right_idx += 1
        else:                           # impares → izquierda
            if left_idx >= 0:
                hyetograph[left_idx] = depth
                left_idx -= 1

    # ── 4. Construir DataFrame ──
    times_min = np.arange(0, duration_min, dt_min)
    times_str = [f"{int(t // 60)}:{int(t % 60):02d}" for t in times_min]

    df = pd.DataFrame({
        'Offset_Min':     times_min,
        'Time_Str':       times_str,
        'Block_Depth_mm': hyetograph,
    })

    # Intensidad equivalente para SWMM (formato INTENSITY en mm/h)
    # mm/bloque ÷ dt_min [min] × 60 [min/h]  →  mm/h
    df['Intensity_mm_h'] = df['Block_Depth_mm'] * (60.0 / dt_min)

    return df


# =============================================================================
# MODIFICACIÓN DEL ARCHIVO INP
# =============================================================================

def generate_inp_file(base_content: str, tr: int,
                      hyetograph_df: pd.DataFrame,
                      output_path: Path) -> Path:
    """
    Genera un archivo .inp de SWMM inyectando el hietograma calculado.

    Estrategia:
      1. Reemplaza globalmente "TORMENTA_COLEGIO_TR{base}" →
                               "TORMENTA_COLEGIO_TR{tr}"
         (actualiza [SUBCATCHMENTS] y [RAINGAGES])
      2. Actualiza la fuente en [RAINGAGES] → TIMESERIES COLEGIO_TR{tr}
      3. Elimina la serie temporal antigua y agrega la nueva en [TIMESERIES]
      4. Actualiza el título [TITLE]
    """
    old_gage_name    = "TORMENTA_COLEGIO_TR25"
    new_gage_name    = f"TORMENTA_COLEGIO_TR{tr}"
    new_ts_name      = f"COLEGIO_TR{tr}"
    base_tr          = config.BASE_INP_TR
    old_ts_name_pfx  = f"COLEGIO_TR{base_tr}"

    # ── 1. Renombrar el pluviómetro globalmente ──
    content = base_content.replace(old_gage_name, new_gage_name)

    # ── 2. Actualizar [TITLE] ──
    if "[TITLE]" in content:
        content = re.sub(
            r'\[TITLE\]\n.*',
            f'[TITLE]\nAnalisis Riesgo - TR {tr} Anios',
            content, count=1
        )
    else:
        content = f"[TITLE]\nAnalisis Riesgo - TR {tr} Anios\n\n" + content

    # ── 3. Procesar línea a línea ──
    lines     = content.splitlines()
    new_lines = []
    in_raingages   = False
    in_timeseries  = False

    for line in lines:
        stripped = line.strip()

        # Detección de sección
        if stripped.startswith("["):
            if stripped.startswith("[RAINGAGES]"):
                in_raingages, in_timeseries = True, False
            elif stripped.startswith("[TIMESERIES]"):
                in_raingages, in_timeseries = False, True
                new_lines.append(line)
                continue
            else:
                in_raingages, in_timeseries = False, False

        if in_raingages and new_gage_name in line and ";" not in line:
            # Reconstruir con el nombre de serie correcto
            new_lines.append(
                f"{new_gage_name} INTENSITY 0:05     1.0      TIMESERIES {new_ts_name}"
            )
        elif in_timeseries:
            # Eliminar la serie base y cualquier duplicado de la nueva
            is_old = stripped.startswith(old_ts_name_pfx) and ";" not in stripped
            is_new = stripped.startswith(new_ts_name)      and ";" not in stripped
            if is_old or is_new:
                continue
            new_lines.append(line)
        else:
            new_lines.append(line)

    # ── 4. Agregar la nueva serie temporal ──
    new_lines.append("\n[TIMESERIES]")
    new_lines.append(
        f";;Generado para TR={tr} años — IDF DAC Aeropuerto (ln)"
    )
    new_lines.append(
        f";;Ecuacion: I[mm/min] = (13.9378*ln(T)+40.7176)/(35.5037+t)^0.9997"
    )
    new_lines.append(";;Name           Date       Time       Value")
    new_lines.append(";;-------------- ---------- ---------- ----------")

    for _, row in hyetograph_df.iterrows():
        val = f"{row['Intensity_mm_h']:.4f}"   # 4 decimales para mayor precisión
        new_lines.append(f"{new_ts_name:<16}           {row['Time_Str']:<10} {val}")

    output_path.write_text("\n".join(new_lines), encoding='latin-1', errors='replace')
    return output_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ── Modo validación ──
    if "--validate" in sys.argv:
        validate_against_table()
        sys.exit(0)

    try:
        print("=" * 60)
        print("rut_22 — Generador de Escenarios SWMM")
        print("IDF: DAC Aeropuerto | Método: Bloques Alternos")
        print("=" * 60)

        print(f"\nLeyendo archivo base: {BASE_INP_PATH}")
        base_content = BASE_INP_PATH.read_text(encoding='latin-1')

        print(f"Períodos de retorno : {RETURN_PERIODS}")
        print(f"Duración            : {DURATION_MIN} min")
        print(f"Paso de tiempo      : {TIME_STEP_MIN} min")
        print(f"Directorio salida   : {OUTPUT_DIR}\n")

        # Tabla de resumen rápida antes de generar
        print(f"{'TR':>5} | {'P_total(mm)':>12} | {'I_max(mm/h)':>12}")
        print("-" * 35)

        summary_data = []

        for tr in RETURN_PERIODS:
            df         = generate_alternating_block_hyetograph(tr, DURATION_MIN, TIME_STEP_MIN)
            total_depth  = df['Block_Depth_mm'].sum()
            max_intensity = df['Intensity_mm_h'].max()

            print(f"{tr:>5} | {total_depth:>12.2f} | {max_intensity:>12.2f}")

            out_name = f"COLEGIO_TR{tr:03d}.inp"
            out_path = OUTPUT_DIR / out_name

            if tr == config.BASE_INP_TR:
                # Para el TR base se copia el archivo directamente
                out_path.write_text(base_content, encoding='latin-1', errors='replace')
                print(f"       → Copiado (TR base): {out_name}")
            else:
                generate_inp_file(base_content, tr, df, out_path)
                print(f"       → Generado: {out_name}")

            summary_data.append({
                'TR':               tr,
                'File':             str(out_path),
                'Total_Depth_mm':   round(total_depth, 3),
                'Max_Intensity_mm_h': round(max_intensity, 3),
            })

        # Guardar CSV de resumen
        summary_csv = OUTPUT_DIR / "scenarios_summary.csv"
        pd.DataFrame(summary_data).to_csv(summary_csv, index=False)

        print(f"\n✓ Generación completa. Resumen guardado en: {summary_csv}")
        print("\nPara verificar los valores contra el Cuadro A4 ejecute:")
        print("  python rut_22_scenario_generator.py --validate\n")

    except Exception:
        import traceback
        print("\nERROR CRÍTICO en rut_22:")
        traceback.print_exc()
        sys.exit(1)