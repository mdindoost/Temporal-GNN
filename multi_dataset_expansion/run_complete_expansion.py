#!/usr/bin/env python3
"""
Master Execution Script - Multi-Dataset Expansion
Runs all phases of the dataset expansion and competitor comparison
"""

import os
import sys
import subprocess
from datetime import datetime

def run_phase(phase_name, script_path, description):
    """Run a single phase and handle errors"""
    
    print(f"\n🚀 STARTING {phase_name}")
    print("=" * 50)
    print(f"Description: {description}")
    print(f"Script: {script_path}")
    
    try:
        # Change to script directory
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        
        if script_dir:
            original_dir = os.getcwd()
            os.chdir(script_dir)
        
        # Run script
        if script_path.endswith('.py'):
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, text=True, timeout=600)
        else:
            result = subprocess.run(['bash', script_name], 
                                  capture_output=True, text=True, timeout=600)
        
        # Restore directory
        if script_dir:
            os.chdir(original_dir)
        
        # Check result
        if result.returncode == 0:
            print(f"✅ {phase_name} COMPLETED SUCCESSFULLY!")
            print("Output:")
            print(result.stdout)
            return True
        else:
            print(f"❌ {phase_name} FAILED!")
            print("Error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {phase_name} TIMED OUT!")
        return False
    except Exception as e:
        print(f"💥 {phase_name} CRASHED: {e}")
        return False

def main():
    """Main execution orchestrator"""
    
    print("🎯 MULTI-DATASET EXPANSION - COMPLETE EXECUTION")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    
    base_path = "/home/md724/temporal-gnn-project/multi_dataset_expansion"
    
    # Phase execution plan
    phases = [
        {
            'name': 'PHASE 1: Bitcoin OTC Validation',
            'script': f'{base_path}/experiments/phase1_bitcoin_otc_validation.py',
            'description': 'Validate findings on Bitcoin OTC dataset'
        },
        {
            'name': 'PHASE 2: Unified Evaluation',
            'script': f'{base_path}/experiments/unified_evaluation_pipeline.py', 
            'description': 'Run comprehensive competitor comparison'
        }
    ]
    
    # Execute phases
    results = []
    
    for phase in phases:
        success = run_phase(
            phase['name'], 
            phase['script'], 
            phase['description']
        )
        results.append((phase['name'], success))
    
    # Summary
    print(f"\n🎯 EXECUTION SUMMARY")
    print("=" * 40)
    
    for phase_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{phase_name}: {status}")
    
    total_success = all(result[1] for result in results)
    
    if total_success:
        print(f"\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print(f"📊 Ready for paper updates")
        print(f"📁 Results in: {base_path}/results/")
    else:
        print(f"\n⚠️  Some phases failed - check individual outputs")
    
    print(f"\nCompleted at: {datetime.now()}")
    
    return total_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
