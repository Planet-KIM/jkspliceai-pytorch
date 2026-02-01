import subprocess
import sys
import os
import json
import time

def run_spliceai_job(loc, ref, alt, model='10k'):
    """
    Executes SpliceAI in a separate subprocess to ensure GPU memory is fully released
    after the job completes.
    """
    
    # Path to the worker script
    worker_script = os.path.join(os.path.dirname(__file__), 'spliceAI_cli.py')
    
    # Command to run (using the same python interpreter)
    cmd = [
        sys.executable, 
        worker_script,
        '--loc', loc,
        '--ref', ref,
        '--alt', alt,
        '--model', model
    ]
    
    print(f"Starting subprocess: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        # Run the subprocess
        # capture_output=True captures stdout and stderr
        # text=True decodes output as string (Python 3.7+)
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        # Parse output
        output_json = result.stdout.strip()
        data = json.loads(output_json)
        
        duration = time.time() - start_time
        print(f"Subprocess completed in {duration:.2f} seconds")
        
        return data

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse output JSON")
        print(f"Raw output: {result.stdout}")
        return None

if __name__ == "__main__":
    # Example usage
    # This simulates a request handler in Flask
    
    print("--- Simulating Request 1 ---")
    res1 = run_spliceai_job(
        loc="chr1:925952-925952", 
        ref="G", 
        alt="A", 
        model='10k'
    )
    print("Result 1:", res1)
    
    print("\n--- Simulating Request 2 (different model) ---")
    res2 = run_spliceai_job(
        loc="chr1:925952-925952", 
        ref="G", 
        alt="A", 
        model='10k_drop'
    )
    print("Result 2:", res2)
