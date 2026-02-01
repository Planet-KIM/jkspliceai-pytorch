import argparse
import sys
import os
import json
import pandas as pd

# Ensure the package is in the python path if running from script directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jkspliceai_pytorch import spliceAI
from jklib.genome import locus

def run_spliceai_cli():
    parser = argparse.ArgumentParser(description="Run SpliceAI inference via CLI")
    parser.add_argument("--loc", type=str, required=True, help="Genomic locus (e.g., chr1:1000-1000)")
    parser.add_argument("--ref", type=str, required=True, help="Reference allele")
    parser.add_argument("--alt", type=str, required=True, help="Alternate allele")
    parser.add_argument("--model", type=str, default="10k", help="Model name (e.g., 10k, 10k_drop)")
    parser.add_argument("--distance", type=int, default=5000, help="Max distance")
    parser.add_argument("--assembly", type=str, default="hg38", help="Genome assembly")
    
    args = parser.parse_args()

    # Create locus object
    try:
        loc = locus(args.loc)
    except Exception as e:
        print(json.dumps({"error": f"Invalid locus format: {str(e)}"}), file=sys.stdout)
        sys.exit(1)

    try:
        # Run inference
        # todict=True is usually better for JSON serialization, but looking at existing usage, 
        # it returns a DataFrame. We'll convert it manually to be safe.
        df = spliceAI(
            loc=loc, 
            ref=args.ref, 
            alt=args.alt, 
            max_distance=args.distance,
            model=args.model, 
            view=10, 
            assembly=args.assembly, 
            verbose=False, # Suppress stdout logging to keep clean for piping
            todict=False
        )

        # Convert result to JSON string for easy parsing by parent process
        if isinstance(df, pd.DataFrame):
            result_json = df.to_json(orient="records")
        else:
            result_json = json.dumps(df)
            
        print(result_json)
        
    except Exception as e:
        # Print error as JSON to stdout so parent can catch it, or print to stderr
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_spliceai_cli()
