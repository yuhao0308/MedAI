"""
MONAI Label Annotation Setup Script
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotation.monai_label_server import (
    setup_monai_label_server,
    get_3d_slicer_connection_instructions,
    create_monai_label_config
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Set up MONAI Label for annotation"
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        required=True,
        help="Path to BIDS dataset root"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="MONAI Label server port"
    )
    parser.add_argument(
        "--instructions-only",
        action="store_true",
        help="Only print setup instructions"
    )
    
    args = parser.parse_args()
    
    if args.instructions_only:
        print("\n=== MONAI Label Server Setup ===")
        print(setup_monai_label_server(args.bids_root, args.port))
        print("\n=== 3D Slicer Connection Instructions ===")
        print(get_3d_slicer_connection_instructions())
    else:
        # Create configuration
        config = create_monai_label_config(
            bids_root=args.bids_root,
            output_dir=str(Path(args.bids_root) / "derivatives" / "labels")
        )
        
        logger.info("MONAI Label configuration created")
        logger.info(f"BIDS root: {args.bids_root}")
        logger.info(f"Server command:")
        print(setup_monai_label_server(args.bids_root, args.port))


if __name__ == "__main__":
    main()


