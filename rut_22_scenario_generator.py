"""
rut_22_scenario_generator.py
============================
Generates SWMM .inp files for multiple Return Periods (TR) using an IDF equation
and the Alternating Block Method.

IDF Equation (DAC-Aeropuerto P10):
    I(t, T) = (13.9378 * ln(T) + 40.7176) / (35.5037 + t)^0.9997

Settings:
- Duration: 60 minutes
- Time Step: 5 minutes
- Return Periods: 1, 2, 5, 10, 25, 50, 100 years
"""

import math
import pandas as pd
import numpy as np
from pathlib import Path
import config
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_INP_PATH = config.SWMM_FILE
OUTPUT_DIR = config.CODIGOS_DIR / "scenarios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION_MIN = 60
TIME_STEP_MIN = 5
RETURN_PERIODS = [1, 2, 5, 10, 25, 50, 100]
# BASE_TR_YEARS is now in config.BASE_INP_TR

# =============================================================================
# IDF & HYETOGRAPH FUNCTIONS
# =============================================================================

def calculate_intensity(duration_min: float, tr_years: float) -> float:
    """
    Calculates rainfall intensity (mm/h) using the user-provided IDF equation.
    
    Equation: I = (13.9378 * ln(T) + 40.7176) / (35.5037 + t)^0.9997
    
    Args:
        duration_min (t): Duration in minutes.
        tr_years (T): Return period in years.
        
    Returns:
        Intensity in mm/h.
    """
    # Note: The equation likely gives intensity in mm/h directly given the coefficients?
    # Let's verify with the check values provided by the user.
    # TR=5, t=60 -> Target ~40.1 mm total depth? Or intensity? 
    # The table says "Cantidad de precipitación máxima (mm)". So the table is DEPTH (P).
    # The equation says "I(t, T)". Usually I is Intensity.
    # Let's check dimensions.
    # If I is mm/hr: P = I * (t/60).
    # Let's calculate I for TR=5, t=60:
    # Num = 13.9378 * ln(5) + 40.7176 = 13.9378*1.609 + 40.7176 = 22.43 + 40.72 = 63.15
    # Denom = (35.5037 + 60)^0.9997 = 95.5037^0.9997 ≈ 95.36
    # Result = 0.662.  
    # If this is mm/min, then P = 0.662 * 60 = 39.7 mm. -> MATCHES TABLE (40.1).
    # CONCLUSION: The equation gives INTENSITY in MM/MIN.
    
    numerator = 13.9378 * math.log(tr_years) + 40.7176
    denominator = (35.5037 + duration_min) ** 0.9997
    intensity_mm_min = numerator / denominator
    
    return intensity_mm_min

def generate_alternating_block_hyetograph(tr_years, duration_min, dt_min):
    """
    Generates a design hyetograph using the Alternating Block Method.
    
    Returns:
        DataFrame with columns ['Time', 'Intensity_mm_h', 'Precip_mm']
        where Time is the block start time (0, 5, 10...)
    """
    n_blocks = int(duration_min / dt_min)
    
    # 1. Calculate Intensity-Duration-Frequency curve for multiples of dt
    # We need Cumulative Depth P(t) = I(t) * t
    durations = np.arange(dt_min, duration_min + dt_min, dt_min) # [5, 10, ..., 60]
    intensities = [calculate_intensity(d, tr_years) for d in durations] # mm/min
    depths = [i * d for i, d in zip(intensities, durations)] # mm (Cumulative)
    
    # 2. Calculate Incremental Depth (Block depth)
    # Block 1 (0-5 avg): Depth(5)
    # Block 2 (5-10 avg): Depth(10) - Depth(5)
    cumulative_depths = [0] + depths
    incremental_depths = np.diff(cumulative_depths) # mm per block
    
    # 3. Sort Scale (Blocks)
    # The Alternating Block Method reorders these increments:
    # Max in center, then alternating left, right, left...
    
    # Sort descending
    sorted_blocks = np.sort(incremental_depths)[::-1]
    
    # Reorder
    hyetograph = [0.0] * n_blocks
    center_idx = n_blocks // 2
    
    # Place max at center (or slightly offset depending on odd/even)
    # Standard approach:
    # 5, 3, 1, 2, 4, 6 (Example order indices)
    
    left_idx = center_idx - 1
    right_idx = center_idx
    
    for i, block_depth in enumerate(sorted_blocks):
        if i % 2 == 0:
            # Place at center/right
            if right_idx < n_blocks:
                hyetograph[right_idx] = block_depth
                right_idx += 1
        else:
            # Place left
            if left_idx >= 0:
                hyetograph[left_idx] = block_depth
                left_idx -= 1
                
    # Create DataFrame
    times_min = np.arange(0, duration_min, dt_min)
    times_str = [f"{int(t//60)}:{int(t%60):02d}" for t in times_min]
    
    df = pd.DataFrame({
        'Offset_Min': times_min,
        'Time_Str': times_str,
        'Block_Depth_mm': hyetograph
    })
    
    # Convert to Intensity (mm/h) for reporting, though SWMM usually takes Intensity or Volume
    # SWMM TIMESERIES format usually takes: Date Time Value
    # If format is INTENSITY, Value = mm/hr. If VOLUME, Value = mm (or rainfall depth).
    # The template INP says: [RAINGAGES] ... Format=INTENSITY Interval=0:05
    # IF format is INTENSITY, we must convert block depth (mm) in 5 min to mm/hr.
    # I_mm_h = Depth_mm * (60 / 5)
    
    df['Intensity_mm_h'] = df['Block_Depth_mm'] * (60 / dt_min)
    
    return df

# =============================================================================
# INP MODIFICATION
# =============================================================================

def generate_inp_file(base_content, tr, hyetograph_df, output_path):
    """
    Creates a new INP file by injecting the generated hyetograph.
    Assumes the base INP has a [TIMESERIES] section.
    
    Update strategy:
    1. Rename "TORMENTA_COLEGIO_TR25" to "TORMENTA_COLEGIO_TR{tr}" globally.
       This ensures [SUBCATCHMENTS] reference the correct gage.
    2. Update [RAINGAGES] to point to the new Timeseries "COLEGIO_TR{tr}".
    3. Append the new [TIMESERIES] "COLEGIO_TR{tr}".
    """
    # Define names
    old_gage_name = "TORMENTA_COLEGIO_TR25"
    new_gage_name = f"TORMENTA_COLEGIO_TR{tr}"
    new_ts_name = f"COLEGIO_TR{tr}"
    
    # 1. Global Replacement of Gage Name (Updates [SUBCATCHMENTS] and [RAINGAGES] name)
    # We do this on the raw content first.
    content = base_content.replace(old_gage_name, new_gage_name)
    
    # 1.1 Update [TITLE] if present, or prepend it
    # We look for "[TITLE]" and replace the lines following it, or inserting new ones.
    # Simple approach: Replace comments or just ensure it's there.
    # We'll just replace the first few lines if they are [TITLE] section.
    
    # Actually, simpler: Use regex or string replace to inject the title.
    if "[TITLE]" in content:
        # Replace the first line of title (usually user defined)
        # Use regex to find [TITLE]\n...
        import re
        content = re.sub(r'\[TITLE\]\n.*', f'[TITLE]\nAnalisis Riesgo - TR {tr} Anios', content, count=1)
    else:
        # Prepend [TITLE]
        content = f"[TITLE]\nAnalisis Riesgo - TR {tr} Anios\n\n" + content
    
    # 2. Process line by line to update the [RAINGAGES] source and append TS
    lines = content.splitlines()
    new_lines = []
    
    in_raingages = False
    in_timeseries = False
    
    # Identify the base TS name to remove. 
    # The base file is TR25, so the TS name usually is COLEGIO_TR25.
    # We can infer it or just filter it out if found.
    # Assuming config.BASE_INP_TR is available and correct (25)
    base_tr = config.BASE_INP_TR
    old_ts_name_prefix = f"COLEGIO_TR{base_tr}"
    
    for line in lines:
        stripped = line.strip()
        
        # Detect start of a section
        if stripped.startswith("["):
            if stripped.startswith("[RAINGAGES]"):
                in_raingages = True
                in_timeseries = False
            elif stripped.startswith("[TIMESERIES]"):
                in_raingages = False
                in_timeseries = True
                new_lines.append(line) # Keep the header
                continue 
            else:
                # Any other section
                in_raingages = False
                in_timeseries = False
        
        # If we are in [RAINGAGES], we need to update the Source.
        if in_raingages and new_gage_name in line and ";" not in line:
            # We construct the line explicitly to ensure correct format
            new_line = f"{new_gage_name} INTENSITY 0:05     1.0      TIMESERIES {new_ts_name}"
            new_lines.append(new_line)
        elif in_timeseries:
            # logic to filter out OLD rainfall series but keep others
            # We must also filter out the NEW series name if it already exists in the file (to avoid duplicates)
            
            is_base_ts = stripped.startswith(old_ts_name_prefix) and ";" not in stripped
            is_new_ts = stripped.startswith(new_ts_name) and ";" not in stripped
            
            if is_base_ts or is_new_ts:
                continue
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # 3. Append new [TIMESERIES]
    new_lines.append("\n[TIMESERIES]")
    new_lines.append(f";;Generated for Return Period {tr} years (IDF: DAC-Aeropuerto P10)")
    new_lines.append(";;Name           Date       Time       Value")
    new_lines.append(";;-------------- ---------- ---------- ----------")
    
    for _, row in hyetograph_df.iterrows():
        t_str = row['Time_Str']
        val = f"{row['Intensity_mm_h']:.2f}"
        new_lines.append(f"{new_ts_name:<16}           {t_str:<10} {val}")
        
    # Write file
    with open(output_path, 'w', encoding='latin-1', errors='replace') as f:
        f.write("\n".join(new_lines))
    
    return output_path

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        print("Reading base INP file...")
        with open(BASE_INP_PATH, 'r', encoding='latin-1') as f:
            base_content = f.read()
            
        print(f"Generating scenarios for TR: {RETURN_PERIODS}")
        print(f"Duration: {DURATION_MIN} min, Step: {TIME_STEP_MIN} min")
        print(f"Output Directory: {OUTPUT_DIR}")
        
        summary_data = []
        
        for tr in RETURN_PERIODS:
            print(f"\nProcessing TR={tr} years...")
            
            if tr == config.BASE_INP_TR:
                print(f"  TR {tr} detected (Same as Base INP). Skipping generation/injection to avoid duplication.")
                # Verify if base content already has the timeseries (it should)
                # Just write the base content to the output file
                out_name = f"COLEGIO_TR{tr:03d}.inp"
                out_path = OUTPUT_DIR / out_name
                with open(out_path, 'w', encoding='latin-1', errors='replace') as f:
                    f.write(base_content)
                print(f"  Copied Base File to: {out_name}")
                
                # Keep variables for summary statistics
                df = generate_alternating_block_hyetograph(tr, DURATION_MIN, TIME_STEP_MIN)
                total_depth = df['Block_Depth_mm'].sum()
                max_intensity = df['Intensity_mm_h'].max()
                
            else:
                # 1. Generate Hyetograph
                df = generate_alternating_block_hyetograph(tr, DURATION_MIN, TIME_STEP_MIN)
                
                total_depth = df['Block_Depth_mm'].sum()
                max_intensity = df['Intensity_mm_h'].max()
                print(f"  Total Depth: {total_depth:.2f} mm")
                print(f"  Max Intensity: {max_intensity:.2f} mm/h")
                
                # 2. Generate INP
                out_name = f"COLEGIO_TR{tr:03d}.inp"
                out_path = OUTPUT_DIR / out_name
                generate_inp_file(base_content, tr, df, out_path)
                print(f"  Saved: {out_name}")
            
            summary_data.append({
                'TR': tr,
                'File': str(out_path),
                'Total_Depth_mm': total_depth,
                'Max_Intensity_mm_h': max_intensity
            })
            
        print("\nGeneration Complete.")
        
        # Save summary CSV for rut_20 to use easily if needed
        pd.DataFrame(summary_data).to_csv(OUTPUT_DIR / "scenarios_summary.csv", index=False)
        
    except Exception as e:
        import traceback
        print("\nCRITICAL ERROR in rut_22:")
        traceback.print_exc()
        sys.exit(1)
