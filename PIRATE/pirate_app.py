"""
Main Streamlit application file.
"""

import modules

choice = modules.main_page()

if choice == 'Basic Sequence Analysis':

    modules.basic_sequence_analysis()

if choice == 'PIRATE Directed Evolution':

    modules.directed_evolution()
