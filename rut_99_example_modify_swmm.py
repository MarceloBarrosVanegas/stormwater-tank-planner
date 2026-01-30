
import config
config.setup_sys_path()
from rut_14_swmm_modifier import SWMMModifier
import os

def run_example():
    # Define paths
    # Replace with your actual paths or uses a temporary test file
    inp_file = r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\modelos\modelo_base.inp" # Example path
    
    if not os.path.exists(inp_file):
        print(f"File not found: {inp_file}. Please check the path.")
        return

    # Initialize Modifier
    print(f"Loading {inp_file}...")
    modifier = SWMMModifier(inp_file)

    # 1. Define modifications
    # Example: Change T-100 to 1.5m Circular, and T-101 to 2.0x1.5m Rectangular
    
    # Simulating data coming from a GeoDataFrame
    links  = ['T-100', 'T-101']
    shapes = ['CIRCULAR', 'RECT_CLOSED']
    geoms  = [
        [1.5, 0, 0, 0],   # For T-100
        [1.5, 2.0, 0, 0]  # For T-101 (Height, Width)
    ]

    # 2. Apply modifications
    print("Applying xsection modifications...")
    count = modifier.modify_xsections(links, shapes, geoms)
    print(f"Modified {count} cross-sections.")

    # 3. Save changes
    output_file = inp_file.replace(".inp", "_mod.inp")
    modifier.save(output_file)
    print(f"Saved modified model to {output_file}")

if __name__ == "__main__":
    run_example()
