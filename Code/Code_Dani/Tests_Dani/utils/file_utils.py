"""
File and directory management utilities

Author: Quantum ML Implementation
Date: November 2025
"""

import os
import os.path as osp


def create_output_directories():
    """
    Create organized directory structure for results
    
    Structure:
    Output/
        Classical/
        Fourier/
        QAOA/
    
    Returns:
        base_dir: Path to base output directory
    """
    base_dir = '../Output'
    methods = ['Classical', 'Fourier', 'QAOA']
    
    for method in methods:
        method_dir = osp.join(base_dir, method)
        os.makedirs(method_dir, exist_ok=True)
    
    print(f"Created output directories in: {base_dir}")
    return base_dir
