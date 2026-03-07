import pandas as pd
from pathlib import Path
from rut_15_dashboard import EvolutionDashboardGenerator

# Load data
df = pd.read_csv('optimization_results/sequence_tracking.csv')
print(f'Data loaded: {len(df)} rows')
print(f'Columns: {list(df.columns)}')

# Create dashboard
dashboard = EvolutionDashboardGenerator(df, Path('optimization_results'))
dashboard.plot_efficiency_by_tank()
dashboard.plot_outfall_reduction_by_tank()
print('Done!')
