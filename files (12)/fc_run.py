#!/usr/bin/env python3
"""
fc_run.py — Entry point for Bhuj SFF Case 7 GUI
Usage: python fc_run.py
Files: fc_run.py (this), fc_app.py (GUI), fc_physics.py (engine), fc_plot_tools.py (style editor)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fc_app import main
if __name__ == '__main__':
    print("Starting Bhuj SFF Case 7 GUI...")
    print("Files needed: fc_run.py, fc_app.py, fc_physics.py, fc_plot_tools.py")
    main()
