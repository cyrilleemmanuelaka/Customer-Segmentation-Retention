"""
Utility functions for customer segmentation project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_visualization():
    """Setup matplotlib and seaborn parameters"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"📊 {title}")
    print("="*60)