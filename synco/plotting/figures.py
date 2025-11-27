# synco/plotting/figures.py

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////
# MAIN SYNCO FIGURE MODULE
# Orchestrates load -> prepare -> plot for the main figures
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////

import os
import logging
from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .distributions import make_pred_distribution_plots as make_dist_plots
from .performance import make_performance_plots as make_perf_plots

logger = logging.getLogger(__name__)

