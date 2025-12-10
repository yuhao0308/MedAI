#!/usr/bin/env python3
"""
Script to execute full pipeline and capture all output for documentation
Used for Phase 0 Week 1 testing and issue documentation
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline_with_capture(config_path: str, output_file: str):
    """
    Run full pipeline and capture all output to file
    
    Args:
        config_path: Path to pipeline config file
        output_file: Path to save captured output
    """
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    
    cmd = [
        sys.executable,
        str(scripts_dir / 'run_pipeline.py'),
        'full',
        '--config', config_path
    ]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    logger.info(f"Output will be saved to: {output_file}")
    
    start_time = datetime.now()
    
    try:
        # Run command and capture both stdout and stderr
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Prepare output
        output_data = {
            'command': ' '.join(cmd),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Also save human-readable version
        readable_file = output_file.replace('.json', '_readable.txt')
        with open(readable_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FULL PIPELINE EXECUTION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start Time: {start_time.isoformat()}\n")
            f.write(f"End Time: {end_time.isoformat()}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write(f"Success: {result.returncode == 0}\n\n")
            f.write("="*80 + "\n")
            f.write("STDOUT\n")
            f.write("="*80 + "\n\n")
            f.write(result.stdout)
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write("STDERR\n")
            f.write("="*80 + "\n\n")
            f.write(result.stderr)
        
        logger.info(f"Execution completed. Return code: {result.returncode}")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"Readable output saved to: {readable_file}")
        
        return output_data
        
    except subprocess.TimeoutExpired:
        logger.error("Pipeline execution timed out after 1 hour")
        return {
            'command': ' '.join(cmd),
            'start_time': start_time.isoformat(),
            'end_time': None,
            'duration_seconds': None,
            'return_code': None,
            'stdout': None,
            'stderr': 'Execution timed out after 1 hour',
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}")
        return {
            'command': ' '.join(cmd),
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'return_code': None,
            'stdout': None,
            'stderr': str(e),
            'success': False,
            'error': str(e)
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Execute full pipeline and capture output for testing'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to pipeline config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: docs/test_execution_output.json)'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        docs_dir = Path(__file__).parent.parent / 'docs'
        docs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = docs_dir / f'test_execution_output_{timestamp}.json'
    
    # Run pipeline
    result = run_pipeline_with_capture(str(config_path), str(output_file))
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Success: {result['success']}")
    print(f"Return Code: {result.get('return_code', 'N/A')}")
    if result.get('duration_seconds'):
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
    print(f"Output saved to: {output_file}")
    print("="*80)
    
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    import argparse
    main()


