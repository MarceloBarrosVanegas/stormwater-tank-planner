from pyswmm import Simulation
import os

baseline_inp = 'optimization_results/00_Baseline/scenarios/COLEGIO_TR025.inp'
print(f'Ejecutando baseline: {baseline_inp}')

sim = Simulation(baseline_inp)
sim.execute()
print('Baseline ejecutado exitosamente')
