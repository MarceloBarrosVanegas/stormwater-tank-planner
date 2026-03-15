#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import traceback

sys.path.insert(0, '.')

# Capturar todo el output
sys.stdout = open('validator_output.log', 'w', encoding='utf-8')
sys.stderr = sys.stdout

try:
    print("Starting validator...")
    import config
    print("Config imported")
    config.setup_sys_path()
    print("Config setup done")
    
    from rut_30_cross_tr_validator import CrossTRValidator
    print("CrossTRValidator imported")
    
    validator = CrossTRValidator()
    print("Validator created")
    print(f"Baselines: {validator.baseline_metrics}")
    
    solution_path = __import__('pathlib').Path(config.CODIGOS_DIR) / "optimization_results" / "Seq_Iter_04" / "model_Seq_Iter_04.inp"
    if solution_path.exists():
        print(f"\nComparing solution: {solution_path}")
        validator.compare_solution(solution_path, iteration_name="Seq_Iter_04")
        print("Comparison done")
    else:
        print(f"Solution not found: {solution_path}")
    
    print("\nAll done!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
