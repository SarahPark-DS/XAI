import os
import subprocess
import sys

def run_deepxplore(transformation, weight_diff=1.0, weight_nc=0.1,
                   step=0.05, seeds=10, grad_iterations=20, threshold=0.5):
    """Run DeepXplore with given parameters."""
    cmd = [
        sys.executable, 'gen_diff.py',
        transformation,
        str(weight_diff),
        str(weight_nc),
        str(step),
        str(seeds),
        str(grad_iterations),
        str(threshold)
    ]
    print(f'\n{"="*50}')
    print(f'Running DeepXplore with transformation: {transformation}')
    print(f'Command: {" ".join(cmd)}')
    print(f'{"="*50}\n')
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode

def main():
    # 각 transformation으로 DeepXplore 실행 (seeds=10으로 빠르게 데모)
    transformations = ['light', 'occl', 'blackout']
    
    for transformation in transformations:
        returncode = run_deepxplore(transformation, seeds=10, grad_iterations=20)
        if returncode != 0:
            print(f'Error running {transformation}')
            sys.exit(1)
    
    print('\n' + '='*50)
    print('DeepXplore demo complete!')
    print('Results saved in generated_inputs/ directory')
    print('Visualizations saved in ../results/ directory')
    print('='*50)

if __name__ == '__main__':
    main()