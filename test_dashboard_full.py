import pandas as pd
from pathlib import Path
from rut_15_dashboard import EvolutionDashboardGenerator

# Load data
df = pd.read_csv('optimization_results/sequence_tracking.csv')
print(f'Data loaded: {len(df)} rows')

# Create dashboard and generate ALL plots
dashboard = EvolutionDashboardGenerator(df, Path('optimization_results'))
dashboard.generate_all()
print('All dashboard figures generated!')
