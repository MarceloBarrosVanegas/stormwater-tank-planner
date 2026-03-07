import pandas as pd
from pathlib import Path
from rut_15_dashboard import EvolutionDashboardGenerator

print("Loading data...")
df = pd.read_csv('optimization_results/sequence_tracking.csv')
print("Initializing generator...")
dash_gen = EvolutionDashboardGenerator(df, Path('optimization_results/test_dir'))
print("Generating plot outfall...")
dash_gen.plot_outfall_reduction_by_tank()
print("Generating plot system util...")
dash_gen.plot_system_utilization_by_tank()
print("Generating plot_flooding_flow_by_tank...")
dash_gen.plot_flooding_flow_by_tank()
print("Done.")
