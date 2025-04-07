# -*- coding: utf-8 -*-
"""

@authors: Samuel Robbins, Chelsea Mackaman-Lofland

"""

# Data Handling/Visualization Libraries
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from collections import defaultdict

# Path Libraries
import os
import pathlib
from pathlib import Path

from typing import Tuple, Dict, List, Union, Optional


# Notebook set up
sns.set(style='white')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams['font.family'] = 'Arial'  # Set default font family
matplotlib.rcParams['font.size'] = 10         # Set default font size

# Shades of Grey
GREY10 = "#1a1a1a"
GREY30 = "#4d4d4d"
GREY40 = "#666666"
GREY50 = "#7f7f7f"
GREY60 = "#999999"
GREY75 = "#bfbfbf"
GREY91 = "#e8e8e8"
GREY98 = "#fafafa"

class SingleSampleModel:
    def __init__(self, file_name: str, sample_name: str, encoding: str = 'latin-1'):
        self.file_name = file_name
        self.sample_name = sample_name
        self.encoding = encoding
        self.path_dict = None
        self.constraints = None
        self.envelope_dict = None
        self.load_data()

    def load_data(self):
        """Loads and processes the data from the file."""
        with open(self.file_name, encoding=self.encoding) as f:
            txt_data_all = f.readlines()

        first_word = [line.split()[0] for line in txt_data_all]

        # Locate key indices for different sections
        envelopesIndex = first_word.index('Envelopes')
        meanPathIndex = first_word.index('Weighted')
        head_length = first_word.index('Fit')

        # Process constraints
        if str.lower(first_word[0]) == 'constraints':
            constraintsIndex = first_word.index('Number')

            num_constraints = (envelopesIndex - 1) - (constraintsIndex + 1)
            self.constraints = np.genfromtxt(
                self.file_name, delimiter='\t', skip_header=2, max_rows=num_constraints, usecols=(1, 2, 3, 4)
            )

        # Process envelopes
        self.envelope_dict = {}
        num_envelopes = (first_word.index('Summaries') - 1) - envelopesIndex
        for i in range(1, num_envelopes + 1):
            envelope_data = txt_data_all[envelopesIndex + i].split('\t')
            envelope_data_clean = [entry.strip() for entry in envelope_data if entry.strip()]  # Remove empty and whitespace-only entries
            envelope_label = envelope_data_clean[0]

            # Only process non-empty values after the label
            self.envelope_dict[envelope_label] = [float(value) for value in envelope_data_clean[1:] if value]

        # Check for acceptable paths only
        if num_envelopes == 3 and 'Acc' in txt_data_all[envelopesIndex + i].split('\t')[0]:
            print('Note that this sample only has acceptable paths.')

        # Identify indices for sample inputs
        grain_headings = first_word.index('Individual')
        sample_grains = txt_data_all[grain_headings+1].split('\t')
        sample_grain_stats = txt_data_all[grain_headings+2].split('\t')

        sample_grains_cleaned = [
                item.replace('\n', '') if '\n' in item else item for item in sample_grains
            ]

        sample_grain_stats_cleaned = [
                item.replace('\n', '') if '\n' in item else item for item in sample_grain_stats
            ]

        sample_grains_cleaned = [s.replace(' ', '_') if s else s for s in sample_grains_cleaned]
        sample_grain_stats_cleaned = [s.replace(' ', '_') if s else s for s in sample_grain_stats_cleaned]
        
        # Process path data
        txt_data = txt_data_all[head_length - 1:]
        raw_data = [line.split('\t') for line in txt_data[1:]]
        data_headers = raw_data[0]
        path_data = raw_data[1:]
        
        # Identify user-specified constraints in path data
        con_index = [idx for idx, val in enumerate(data_headers) if 'con' in val.lower()]
        data_start = path_data[0].index('Time (Ma)')
        data_cleaned = [
            [
                item.replace('\n', '') if '\n' in item else item
                for item in line
            ]
            for line in path_data
        ]

        # Organize path data into dictionary and convert strings to floats
        self.path_dict = defaultdict(dict)
        for path in data_cleaned:
            path_id = path[0]
            path_type = str.lower(path[data_start])

            # Determine the data type (time, temp, or depth)
            if 'time (ma)' in path_type:
                data_type_key = 'time'
            elif 'temp' in path_type:
                data_type_key = 'temp'
            elif 'depth' in path_type:
                data_type_key = 'depth'
            else:
                continue

            # Convert the data entries to floats and assign to path_dict
            main_data = [float(value) for value in path[data_start + 1:] if value]
            self.path_dict[path_id][data_type_key] = main_data

            if data_type_key == 'time':
                self.path_dict[path_id]['stage'] = path[1]
                for i in range(2, data_start):
                    grain_id = sample_grains_cleaned[i]
                    grain_stat1 = sample_grain_stats_cleaned[i]

                    # Initialize or update the dictionary for the grain_id
                    if grain_id not in self.path_dict[path_id]:
                        self.path_dict[path_id][grain_id] = {}
                    
                    if grain_stat1 in ('corr._age_(Ma)','age_(Ma)'):
                        self.path_dict[path_id][grain_id]['modeled_age'] = float(path[i]) if path[i] else None
                    else:
                        self.path_dict[path_id][grain_id][grain_stat1] = float(path[i]) if path[i] else None
            else:
                self.path_dict[path_id]['comp_GOF'] = float(path[1]) if path[1] else None
                for i in range(2, data_start):
                    grain_id = sample_grains_cleaned[i]
                    grain_stat2 = data_headers[i]

                    # Initialize or update the dictionary for the grain_id
                    if grain_id not in self.path_dict[path_id]:
                        self.path_dict[path_id][grain_id] = {}
                    
                    if grain_stat2 == 'GOF':
                        self.path_dict[path_id][grain_id]['modeled_age_GOF'] = float(path[i]) if path[i] else None
                    else:
                        self.path_dict[path_id][grain_id][grain_stat2] = float(path[i]) if path[i] else None

            # Convert and store user-specified constraint values as floats
            con_only = [float(val) for idx, val in enumerate(path) if idx in con_index and val]
            self.path_dict[path_id][f'{data_type_key}_con'] = con_only

        # Identify distinct samples from model input parameters
        seen = set()
        self.sample_grain_list = [x for x in sample_grains_cleaned if x and not (x in seen or seen.add(x))]

        # Pull Best Path
        for key, path in self.path_dict.items():
            if 'Best' in key:
                self.best_path = path

    def get_path_data(self) -> Dict[str, Dict[str, Union[List[str], List[float], None]]]:
        """Returns the path data dictionary."""
        return self.path_dict

    def get_constraints(self) -> np.ndarray:
        """Returns the constraints array."""
        return self.constraints

    def get_envelopes(self) -> Dict[str, List[float]]:
        """Returns the envelope data dictionary."""
        return self.envelope_dict

    def identifyPathFamilies(self,
        plot_type: str,
        y_variable: str,
        y_lim: Optional[Tuple[float, float]] = None,
        x_lim: Optional[Tuple[float, float]] = None,
        fig_size: Optional[Tuple[float, float]] = None,
        x_grid_spacing: Optional[float] = None,  
        y_grid_spacing: Optional[float] = None,  
        grid_linewidth: float = 0.5,  
        good_match_color: str = 'goldenrod',
        acc_match_color: str = 'gold',
        showConstraintBoxes: bool = False,
        constraintBoxColor: str = 'red',
        constraintLineColor: str = 'black',
        constraintMarkerStyle: str = 's',
        saveFig: bool = False,
        saveFolder: str = 'Plots',
        savefigFileName: Optional[str] = None,
        c1_x: Optional[Tuple[float, float]] = None,
        c1_y: Optional[Tuple[float, float]] = None,
        c2_x: Optional[Tuple[float, float]] = None,
        c2_y: Optional[Tuple[float, float]] = None,
        c3_x: Optional[Tuple[float, float]] = None,
        c3_y: Optional[Tuple[float, float]] = None,
        c4_x: Optional[Tuple[float, float]] = None,
        c4_y: Optional[Tuple[float, float]] = None,
        c5_x: Optional[Tuple[float, float]] = None,
        c5_y: Optional[Tuple[float, float]] = None,
        c6_x: Optional[Tuple[float, float]] = None,
        c6_y: Optional[Tuple[float, float]] = None,
        c7_x: Optional[Tuple[float, float]] = None,
        c7_y: Optional[Tuple[float, float]] = None,
        c8_x: Optional[Tuple[float, float]] = None,
        c8_y: Optional[Tuple[float, float]] = None,
        c9_x: Optional[Tuple[float, float]] = None,
        c9_y: Optional[Tuple[float, float]] = None,
        c10_x: Optional[Tuple[float, float]] = None,
        c10_y: Optional[Tuple[float, float]] = None,
        c11_x: Optional[Tuple[float, float]] = None,
        c11_y: Optional[Tuple[float, float]] = None,
        c12_x: Optional[Tuple[float, float]] = None,
        c12_y: Optional[Tuple[float, float]] = None,
        c13_x: Optional[Tuple[float, float]] = None,
        c13_y: Optional[Tuple[float, float]] = None,
        c14_x: Optional[Tuple[float, float]] = None,
        c14_y: Optional[Tuple[float, float]] = None,
        c15_x: Optional[Tuple[float, float]] = None,
        c15_y: Optional[Tuple[float, float]] = None
        ) -> List[str]:
        """
        Identifies and visualizes families of thermal history paths that pass through user-specified
        constraints. This function allows users to define up to 15 constraint regions and identifies
        which paths satisfy all specified constraints.

        Parameters
        ----------
        sample : str
            Name of the sample being analyzed.

        plot_type : str
            Type of visualization to create. Options are:
            - 'paths': Shows continuous thermal history paths
            - 'points': Shows only the constraint points

        y_variable : str
            Variable to plot on y-axis. Options are:
            - 'temp': Temperature in °C
            - 'depth': Depth in meters

        y_lim : tuple[float, float], optional
            Custom y-axis limits.
            If None, limits are automatically determined from the data.

        x_lim : tuple[float, float], optional
            Custom x-axis limits.
            If None, limits are automatically determined from the data.

        good_match_color : str, default='goldenrod'
            Color for good-fit paths that satisfy all constraints.

        acc_match_color : str, default='gold'
            Color for acceptable-fit paths that satisfy all constraints.

        showConstraintBoxes : bool, default=False
            If True, displays the original model constraint boxes and connecting lines.

        constraintBoxColor : str, default='red'
            Color of the original constraint box outlines.

        constraintLineColor : str, default='black'
            Color of the lines connecting original constraint boxes.

        constraintMarkerStyle : str, default='s'
            Marker style for original constraint box centers.
            Any valid matplotlib marker style.

        saveFig : bool, default=False
            If True, saves the figure to disk.

        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            Will be created if it doesn't exist.

        savefigFileName : str, optional
            Custom filename for the saved figure. If None and saveFig is True,
            a default filename will be generated.

        c1_x through c15_x : tuple[float, float], optional
            X-axis (time) bounds for constraints 1-15 in the form (max_time, min_time).
            Note the reverse order to match plot direction.
            If None, that constraint is not applied.

        c1_y through c15_y : tuple[float, float], optional
            Y-axis (temperature/depth) bounds for constraints 1-15 in the form (max_value, min_value).
            Note the reverse order to match plot direction.
            If None, that constraint is not applied.

        Returns
        -------
        key_matched_paths : list[str]
            List of path identifiers that satisfy all specified constraints.
            These identifiers can be used to access the path data from the model.

        Notes
        -----
        - The function supports up to 15 user-defined constraints, but you only need to specify up to the ones you want to use.
        - Paths must pass through ALL specified constraints to be considered part of the path family.
        - Constraints are applied in reverse order (youngest to oldest).
        
        """    
        path_dict = self.get_path_data()
        constraints = self.get_constraints()
        
        if y_variable == 'depth':
            y_all = 'depth'
            y_con = 'depth_con'
        
        elif y_variable == 'temp':
            y_all = 'temp'
            y_con = 'temp_con'
            
    
        # Breaking up the dictionary to ensure correct layering for different path groups
        best_path = {}
        good_paths = {}
        acc_paths = {}
    
        for key, path in path_dict.items():
            if 'best' in key.lower():
                best_path[key] = path
            elif 'good' in key.lower():
                good_paths[key] = path
            elif 'acc' in key.lower():
                acc_paths[key] = path
        
        ### ----------------------------------- 
        ### Identifying path families 
        key_matched_paths = []
    
        # Define a list of constraints for x and y - user submitted
        constraints_x = [c1_x, c2_x, c3_x, c4_x, c5_x, c6_x, c7_x, c8_x, c9_x, c10_x, c11_x, c12_x, c13_x, c14_x, c15_x]
        constraints_y = [c1_y, c2_y, c3_y, c4_y, c5_y, c6_y, c7_y, c8_y, c9_y, c10_y, c11_y, c12_y, c13_y, c14_y, c15_y]
    
        # Iterate over all paths and test against constraints 
        for key, path in best_path.items():
            meets_conditions = True
            for i, (constraint_x, constraint_y) in enumerate(zip(constraints_x, constraints_y), start=1):
                if constraint_x or constraint_y:  # Check if either x or y constraint is provided
                    x_condition_met = not constraint_x or (len(path['time_con']) >= (i + 1) and constraint_x[0] >= path['time_con'][-(i + 1)] >= constraint_x[1])
                    y_condition_met = not constraint_y or (len(path[y_con]) >= (i + 1) and constraint_y[0] >= path[y_con][- (i + 1)] >= constraint_y[1])
                    if not (x_condition_met and y_condition_met):
                        meets_conditions = False
                        break
            if meets_conditions:
                key_matched_paths.append(key)
    
        for key, path in good_paths.items():
            meets_conditions = True
            for i, (constraint_x, constraint_y) in enumerate(zip(constraints_x, constraints_y), start=1):
                if constraint_x or constraint_y:  # Check if either x or y constraint is provided
                    x_condition_met = not constraint_x or (len(path['time_con']) >= (i + 1) and constraint_x[0] >= path['time_con'][-(i + 1)] >= constraint_x[1])
                    y_condition_met = not constraint_y or (len(path[y_con]) >= (i + 1) and constraint_y[0] >= path[y_con][- (i + 1)] >= constraint_y[1])
                    if not (x_condition_met and y_condition_met):
                        meets_conditions = False
                        break
            if meets_conditions:
                key_matched_paths.append(key)
    
        for key, path in acc_paths.items():
            meets_conditions = True
            for i, (constraint_x, constraint_y) in enumerate(zip(constraints_x, constraints_y), start=1):
                if constraint_x or constraint_y:  # Check if either x or y constraint is provided
                    x_condition_met = not constraint_x or (len(path['time_con']) >= (i + 1) and constraint_x[0] >= path['time_con'][-(i + 1)] >= constraint_x[1])
                    y_condition_met = not constraint_y or (len(path[y_con]) >= (i + 1) and constraint_y[0] >= path[y_con][- (i + 1)] >= constraint_y[1])
                    if not (x_condition_met and y_condition_met):
                        meets_conditions = False
                        break
            if meets_conditions:
                key_matched_paths.append(key)
        
        
        ### ----------------------------------- 
        ### Figure Set-up
        if fig_size:
            fig_size = fig_size
        else: 
            fig_size = (12,6)

        fig, ax = plt.subplots(1, 1, figsize = fig_size)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    
        ### ----------------------------------- 
        ### Plot the paths
    
        if plot_type == 'paths':
    
            for key, path in acc_paths.items():
                
                # separate out the paths in the path family
                if key in key_matched_paths: 
                    path_color = acc_match_color
                    plt.plot(path['time'],path[y_all], color = path_color, alpha = 1, zorder = 10000)
                    
                else:
                    path_color = 'gainsboro'
                    plt.plot(path['time'],path[y_all], color = path_color, alpha = 1)
    
            for key, path in good_paths.items():
                
                # separate out the paths in the path family
                if key in key_matched_paths: 
                    path_color = good_match_color
                    plt.plot(path['time'],path[y_all], color = path_color, alpha = 1, zorder = 10000)
                    
                else:
                    path_color = 'silver'
                    plt.plot(path['time'],path[y_all], color = path_color, alpha = 1)
            
        if plot_type == 'points':
    
            for key, path in acc_paths.items():
                
                # separate out the paths in the path family
                if key in key_matched_paths: 
                    path_color = acc_match_color
                    plt.scatter(path['time_con'],path[y_con], color = path_color, alpha = 1, marker = 's', 
                                zorder = 1000)
                else:
                    path_color = 'gainsboro'
                    plt.scatter(path['time_con'],path[y_con], alpha = 1, marker = 's',
                                   color=path_color
                               )
    
            for key, path in good_paths.items():
                
                # separate out the paths in the path family
                if key in key_matched_paths: 
                    path_color = good_match_color
                    plt.scatter(path['time_con'],path[y_con], color = path_color, alpha = 1, marker = 's',
                               zorder = 1000)
                else:
                    path_color = 'silver'
                    plt.scatter(path['time_con'],path[y_con], color = path_color, alpha = 1, marker = 's')
                
    
        
        # always plot the best fit line
        for key, path in best_path.items():
            path_color = 'black'
            plt.plot(path['time'],path[y_all], color = path_color, linewidth = 3, zorder = 10000)
    
        ### -----------------------------------    
        ### Plot the constraint boxes
    
        if showConstraintBoxes:
            
            x_midpts = []
            y_midpts = []
            
            for constraint in constraints:
                max_t = constraint[0]
                min_t = constraint[1]
                max_T = constraint[2]
                min_T = constraint[3]
                
                mid_x = constraint[0] - ((constraint[0]-constraint[1])/2)
                mid_y = constraint[2] - ((constraint[2]-constraint[3])/2)
        
                x_midpts.append(mid_x)
                y_midpts.append(mid_y)
                
                height = -(max_T - min_T)
                width = -(max_t - min_t)
                
                ax.add_patch(Rectangle((max_t,max_T),width,height,
                                        edgecolor = constraintBoxColor,
                                        facecolor = None,
                                        lw = 1.5,
                                        fill = False,
                                        zorder = 100000)) #zorder - arbitrarily large number to bring to front
                
            plt.plot(x_midpts,y_midpts, linestyle = '--', color = constraintLineColor)
            plt.scatter(x_midpts,y_midpts, marker = constraintMarkerStyle, color = 'black', s = 75, facecolors = 'none', zorder = 1000)
        
        
        ### ----------------------------------- 
        ### Axes and Spine Customization
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    
        plt.gca().set_xlim(right=0)
        plt.gca().set_ylim(top=0)
    
        ax.spines["left"].set_color('k')
        ax.spines["bottom"].set_color('k')
        ax.spines["right"].set_color('k')
        ax.spines["top"].set_color('k')

        ### Grid Customization -----------------------------
        if x_grid_spacing is not None:  # Changed to explicit None check
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_grid_spacing))
        if y_grid_spacing is not None:  # Changed to explicit None check
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(y_grid_spacing))

        ax.grid(True, axis='x', color=GREY91, linestyle='-', linewidth=grid_linewidth)
        ax.grid(True, axis='y', color=GREY91, linestyle='-', linewidth=grid_linewidth)
    
        ax.xaxis.set_label_position('top')
        
        if y_lim:
            ax.set_ylim(y_lim)
        
        if x_lim:
            ax.set_xlim(x_lim)
    
        ax.set_xlabel('Time (Ma)',
                    labelpad = 8.0,
                    fontname= "Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )
        
        if y_variable == 'depth':
            y_label = 'Depth (m)'
        elif y_variable == 'temp':
            y_label = 'Temperature (ºC)'
        
        ax.set_ylabel(y_label,
                    fontname= "Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )
    
        ax.tick_params(axis="x", length=2, color=GREY91)
        ax.tick_params(axis="y", length=2, color=GREY91)
    
        plt.xticks(fontsize = 10, fontname = 'Arial', color = GREY40)
        plt.yticks(fontsize = 10, fontname = 'Arial', color = GREY40)
        
        
        ### Final Formatting -----------------------------
        title = 'Matched Paths for {} – {}'.format(self.sample_name, plot_type.capitalize())
        plt.title(title,
                  fontsize = 15,
                  fontname = "Arial",
                  color = GREY10,
                  pad = 15)
    
        ### Show and Save Figure -----------------------------
        if saveFig:
            pathlib.Path(saveFolder).mkdir(parents=True, exist_ok=True)
            if savefigFileName:
                filepath = '{}/{}.pdf'.format(saveFolder, savefigFileName)
            else:
                filepath = '{}/{}_{}_HeFty_plots.pdf'.format(saveFolder, self.sample_name, plot_type)
            plt.savefig(filepath, dpi = 'figure', bbox_inches='tight', pad_inches = 0.5)
    
        fig.tight_layout()
        plt.show()
        
        print('There are {} paths that meet these conditions.'.format(len(key_matched_paths)))
        
        return key_matched_paths
    
    def plotSingleSamplePathData(self,
        plot_type: str,
        y_variable: str,
        pathsToPlot: str = 'all',
        envelopesToPlot: Optional[str] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        x_lim: Optional[Tuple[float, float]] = None,
        fig_size: Optional[Tuple[float, float]] = None,
        x_grid_spacing: Optional[float] = None,  
        y_grid_spacing: Optional[float] = None,  
        grid_linewidth: float = 0.5,  
        plotAgeHistogram: bool = False,
        bin_width: int = 5,
        histogramColorPalette: str = 'Dark2',
        stat: str = 'count',
        defaultHeftyStyle: bool = True,
        bestPathColor: str = 'black',
        goodPathColor: str = 'fuchsia',
        accPathColor: str = 'limegreen',
        showConstraintBoxes: bool = True,
        constraintBoxColor: str = 'red',
        constraintLineColor: str = 'black',
        constraintMarkerStyle: str = 's',
        saveFig: bool = False,
        saveFolder: str = 'Plots',
        savefigFileName: Optional[str] = None
        ) -> None:
        """
        Creates visualizations of thermal history paths, including time-temperature or time-depth paths,
        constraint boxes, and optionally age distribution histograms.

        Parameters
        ----------
        plot_type : str
            Type of plot to create. Options are:
            - 'paths': Shows continuous thermal history paths
            - 'points': Shows only the constraint points
            - 'envelopes': Shows the envelope of all paths

        y_variable : str
            Variable to plot on y-axis. Options are:
            - 'temp': Temperature in °C
            - 'depth': Depth in meters

        pathsToPlot : str, default='all'
            Controls which thermal history paths to include. Options are:
            - 'good': Shows only good-fit paths (including best path)
            - 'acc': Shows only acceptable-fit paths
            - 'all': Shows both good and acceptable paths

        envelopesToPlot : str, optional
            When plot_type is 'envelopes', specifies which envelopes to show:
            - 'good': Shows only good-fit envelope
            - 'acc': Shows only acceptable-fit envelope
            - 'both': Shows both envelopes

        y_lim : tuple[float, float], optional
            Custom y-axis limits.
            If None, limits are automatically determined from the data.

        x_lim : tuple[float, float], optional
            Custom x-axis limits.
            If None, limits are automatically determined from the data.
        
        fig_size : tuple[float, float], optional
            Custom dimensions for the figure in inches as (width, height).
            If None when plotAgeHistogram is False: defaults to (16,6)
            If None when plotAgeHistogram is True: defaults to (15,8)

        x_grid_spacing : float, optional
            Spacing between vertical grid lines in Ma.
            If None, uses matplotlib's automatic grid spacing.

        y_grid_spacing : float, optional  
            Spacing between horizontal grid lines in °C or m.
            If None, uses matplotlib's automatic grid spacing.

        grid_linewidth : float, default=0.5
            Width of the grid lines.

        plotAgeHistogram : bool, default=False
            If True, adds a histogram of ages below the main plot.

        bin_width : int, default=5
            Width of bins for the age histogram in Ma.
            Only used when plotAgeHistogram is True.

        histogramColorPalette : str, default='Dark2'
            Name of the matplotlib colormap to use for the histogram.
            Only used when plotAgeHistogram is True.

        stat : str, default='count'
            Statistic to plot in histogram. Options are:
            - 'count': Shows frequency counts
            - 'density': Shows probability density
            - 'probability': Shows probability

        defaultHeftyStyle : bool, default=True
            If True, uses HeFTy-like styling with color gradients based on GOF values.
            If False, uses solid colors specified by bestPathColor, goodPathColor, and accPathColor.

        bestPathColor : str, default='black'
            Color for the best-fit path when defaultHeftyStyle is False.

        goodPathColor : str, default='fuchsia'
            Color for good-fit paths when defaultHeftyStyle is False.

        accPathColor : str, default='limegreen'
            Color for acceptable-fit paths when defaultHeftyStyle is False.

        showConstraintBoxes : bool, default=True
            If True, displays constraint boxes and connecting lines.

        constraintBoxColor : str, default='red'
            Color of the constraint box outlines.

        constraintLineColor : str, default='black'
            Color of the lines connecting constraint boxes.

        constraintMarkerStyle : str, default='s'
            Marker style for constraint box centers.
            Any valid matplotlib marker style.

        saveFig : bool, default=False
            If True, saves the figure to disk.

        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            Will be created if it doesn't exist.

        savefigFileName : str, optional
            Custom filename for the saved figure. If None and saveFig is True,
            a default filename will be generated based on the plot type.

        Returns
        -------
        None
            Function displays the plot and optionally saves it to disk.
        """
        path_dict = self.get_path_data()
        constraints = self.get_constraints()
        envelope_dict = self.get_envelopes()
    
        # Set the type of plot (time/depth)
        if y_variable == 'depth':
            y_all = 'depth'
            y_con = 'depth_con'
        elif y_variable == 'temp':
            y_all = 'temp'
            y_con = 'temp_con'
        else:
            print('That is not a valid y_variable, please input either "temp" or "depth".')
    
        # breaking up the dictionary to ensure correct layering for different path groups
        best_path = {}
        good_paths = {}
        acc_paths = {}

        for key, path in path_dict.items():
            if 'best' in key.lower():
                best_path[key] = path
            elif 'good' in key.lower() and (pathsToPlot in ['all', 'good']):
                good_paths[key] = path
            elif 'acc' in key.lower() and (pathsToPlot in ['all', 'acc']):
                acc_paths[key] = path
    
        ### ----------------------------------- 
        ### Figure Set-up
        if plotAgeHistogram:
            if fig_size:
                fig_size = fig_size
            else: 
                fig_size = (15,8)

            fig, (ax, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, figsize=fig_size, sharex = True)
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:
            if fig_size:
                fig_size = fig_size
            else: 
                fig_size = (16,6)

            fig, ax = plt.subplots(1, 1, figsize = fig_size)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
    
        ### ----------------------------------- 
        ### Plot the Age Histogram
        if plotAgeHistogram:
            constraintHistogramDict = {}

            # Loop through the path dictionary with filtering
            for key, path in path_dict.items():
                # Apply path filtering
                if not ('best' in key.lower() or 
                    ('good' in key.lower() and pathsToPlot in ['all', 'good']) or 
                    ('acc' in key.lower() and pathsToPlot in ['all', 'acc'])):
                    continue

                # Get the time_con list from the path dictionary - only get the "real" constraint points from model
                time_con_list = path.get('time_con', [])

                # Loop through the time_con list and group values by index
                for i, value in enumerate(time_con_list):
                    # Create a new key in the grouped dictionary if it doesn't exist
                    if i not in constraintHistogramDict:
                        constraintHistogramDict[i] = []

                    # Append the value to the corresponding key
                    constraintHistogramDict[i].append(value)
    
            # Plotting histogram - generate colors for all constraint keys
            cmap1 = plt.cm.get_cmap(histogramColorPalette, len(constraintHistogramDict))
            
            # set the bins
            bins = np.arange(x_lim[1], x_lim[0] + bin_width, bin_width)
    
            # Plot histograms for each constraint
            for i, (constraint_index, values) in enumerate(sorted(constraintHistogramDict.items(), reverse=True)):
                if i == 0:
                    label = 'Present Day Conditions'
                elif i == -1:
                    label = f'Constraint Box {len(constraintHistogramDict) }'
                else:
                    label = f'Constraint Box {abs(i) }'
                
                # catch for if a constraint is all 0's e.g. final/present day conditions
                if min(values) == max(values) == 0:
                    pass
                else:
                    sns.histplot(ax=ax1, x=values, stat = stat, bins=bins, kde=True, color=cmap1(i), label=label)
            
            # Adding the legend
            ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    
            # Place y-axis labels on the right side
            ax1.yaxis.set_label_position("right")
            ax1.yaxis.tick_right()
            
            # Axis Label and Settings
            ax1.set_ylabel(stat.title(),
                labelpad = 10,
                fontname= "Arial",
                fontsize=12,
                weight=500,
                color=GREY40,
                rotation = 270
                )
            
            ax1.tick_params(axis='y', length=2, color=GREY91)
        
        ### ----------------------------------- 
        ### Plot the Path Data
        
        # greens colormaps to match HeFTy style
        cmap = LinearSegmentedColormap.from_list('custom cmap', [(0, 'darkolivegreen'), (1, 'lime')])
    
        if plot_type == 'paths':

            if pathsToPlot in ['all', 'acc']:
                for key, path in acc_paths.items():
                    color = cmap(path['comp_GOF']) if defaultHeftyStyle else accPathColor
                    ax.plot(path['time'], path[y_all], color=color, alpha=1)
            
            # Plot good paths if included
            if pathsToPlot in ['all', 'good']:
                for key, path in good_paths.items():
                    color = 'fuchsia' if defaultHeftyStyle else goodPathColor
                    ax.plot(path['time'], path[y_all], color=color, alpha=1)
    
            
            # Plot the color bar for GOF - default style only
            if defaultHeftyStyle and pathsToPlot in ['acc','all']:
                if plotAgeHistogram : 
                    # need to specify placement when histogram is turned on
                    cbar_ax = fig.add_axes([0.97, 0.3, 0.015, 0.55])  # Position of the color bar
                    
                    norm = plt.Normalize(0, 0.5)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
                    
                    cbar.ax.tick_params(axis='y', labelsize=10)
                    
                    cbar.set_label('Acc GOF Value', 
                                   labelpad=15.0, 
                                   fontname="Arial", 
                                   fontsize=12, 
                                   weight=500,
                                   color=GREY40,
                                   rotation=270
                                  )
                else:
                    norm = plt.Normalize(0, 0.5)
                    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax)
    
                    cbar.ax.tick_params(axis='y', labelsize=10, labelcolor=GREY40)
                    for label in cbar.ax.yaxis.get_ticklabels():
                        label.set_fontname('Arial')
    
                    cbar.set_label('Acc GOF Value', 
                                   labelpad = 15.0,
                                   fontname= "Arial",
                                   fontsize=12,
                                   weight=500,
                                   color=GREY40,
                                   rotation = 270
                                   )
            
        if plot_type == 'points':
    
            # Acceptable Paths
            for key, path in acc_paths.items():
                color = cmap(path['comp_GOF']) if defaultHeftyStyle else accPathColor
                ax.scatter(path['time_con'], path[y_con], color=color, alpha=1, marker='s')
    
            # Good Paths
            for key, path in good_paths.items():
                color = 'fuchsia' if defaultHeftyStyle else goodPathColor
                ax.scatter(path['time_con'], path[y_con], color=color, alpha=1, marker='s')
                
            
            # Plot the color bar for GOF - default style only
            if defaultHeftyStyle and pathsToPlot in ['acc','all']:
                if plotAgeHistogram :
                    cbar_ax = fig.add_axes([0.97, 0.3, 0.015, 0.55])  # Position of the color bar
                    
                    norm = plt.Normalize(0, 0.5)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
                    
                    cbar.ax.tick_params(axis='y', labelsize=10)
                    
                    cbar.set_label('Acc GOF Value', 
                                   labelpad=15.0, 
                                   fontname="Arial", 
                                   fontsize=12, 
                                   weight=500,
                                   color=GREY40,
                                   rotation=270
                                  )
                else:
                    norm = plt.Normalize(0, 0.5)
                    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax)
    
                    cbar.ax.tick_params(axis='y', labelsize=10, labelcolor=GREY40)
                    for label in cbar.ax.yaxis.get_ticklabels():
                        label.set_fontname('Arial')
    
                    cbar.set_label('Acc GOF Value', 
                                   labelpad = 15.0,
                                   fontname= "Arial",
                                   fontsize=12,
                                   weight=500,
                                   color=GREY40,
                                   rotation = 270
                                   )
            
        if plot_type == 'envelopes':
            
            # Setting envelope colors based on user input 
            if defaultHeftyStyle:
                goodEnvelopeColor = 'fuchsia'
                accEnvelopeColor = 'lime'
            else:
                goodEnvelopeColor = goodPathColor
                accEnvelopeColor = accPathColor
    
            if y_variable == 'temp':
          
                if envelopesToPlot == 'both':
    
                    # Check to see if there are only acceptable paths
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi temp (C)'], 
                                     envelope_dict['Acc lo temp (C)'], alpha = 0.7, color = accEnvelopeColor)
    
                        print('Note that this sample only has acceptable paths.')
    
                    else:
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi temp (C)'], 
                                     envelope_dict['Acc lo temp (C)'], alpha = 0.7, color = accEnvelopeColor)
    
                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi temp (C)'], 
                                         envelope_dict['Good lo temp (C)'], alpha = 0.9, color = goodEnvelopeColor)
    
                elif envelopesToPlot == 'acc':
                    # Acc Paths
                    ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi temp (C)'], 
                                     envelope_dict['Acc lo temp (C)'], alpha = 0.7, color = accEnvelopeColor)
    
                elif envelopesToPlot == 'good':
    
                    # Verify that there are good paths 
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        print('There are no good paths for this sample. Try plotting the acceptable paths')
    
                    else:
                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi temp (C)'], 
                                         envelope_dict['Good lo temp (C)'], alpha = 0.9, color = goodEnvelopeColor)
                        
            elif y_variable == 'depth':
          
                if envelopesToPlot == 'both':
    
                    # Check to see if there are only acceptable paths
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi depth (m)'], 
                                     envelope_dict['Acc lo depth (m)'], alpha = 0.7, color = accEnvelopeColor)
    
                        print('Note that this sample only has acceptable paths.')
    
                    else:
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi depth (m)'], 
                                     envelope_dict['Acc lo depth (m)'], alpha = 0.7, color = accEnvelopeColor)
    
                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi depth (m)'], 
                                         envelope_dict['Good lo depth (m)'], alpha = 0.9, color = goodEnvelopeColor)
    
                elif envelopesToPlot == 'acc':
                    # Acc Paths
                    ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi depth (m)'], 
                                     envelope_dict['Acc lo depth (m)'], alpha = 0.7, color = accEnvelopeColor)
    
                elif envelopesToPlot == 'good':
    
                    # Verify that there are good paths 
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        print('There are no good paths for this sample. Try plotting the acceptable paths')
    
                    else:
                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi depth (m)'], 
                                         envelope_dict['Good lo depth (m)'], alpha = 0.9, color = goodEnvelopeColor)
                
        # Plot the best fit path         
        for key, path in best_path.items():
            path_color = 'black' if defaultHeftyStyle else bestPathColor
            ax.plot(path['time'],path[y_all], color = path_color, linewidth = 3)
    
        ### -----------------------------------    
        ### Plot the constraint boxes
    
        if showConstraintBoxes and self.constraints is not None:
            
            x_midpts = []
            y_midpts = []
            
            for constraint in constraints:
                max_t = constraint[0]
                min_t = constraint[1]
                max_T = constraint[2]
                min_T = constraint[3]
                
                mid_x = constraint[0] - ((constraint[0]-constraint[1])/2)
                mid_y = constraint[2] - ((constraint[2]-constraint[3])/2)
        
                x_midpts.append(mid_x)
                y_midpts.append(mid_y)
                
                height = -(max_T - min_T)
                width = -(max_t - min_t)
                
                ax.add_patch(Rectangle((max_t,max_T),width,height,
                                        edgecolor = constraintBoxColor,
                                        facecolor = None,
                                        lw = 1.5,
                                        fill = False,
                                        zorder = 100000000)) #zorder - arbitrarily large number to bring to front
                
    
            ax.plot(x_midpts,y_midpts, linestyle = '--', color = constraintLineColor)
            ax.scatter(x_midpts,y_midpts, marker = constraintMarkerStyle , color = 'black', s = 75, facecolors = 'none', zorder = 1000)
    
        ### ----------------------------------- 
        ### Axes and Spine Customization
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    
        ax.spines["left"].set_color('k')
        ax.spines["bottom"].set_color('k')
        ax.spines["right"].set_color('k')
        ax.spines["top"].set_color('k')

        ### Grid Customization -----------------------------
        if x_grid_spacing is not None:  # Explicit None check
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_grid_spacing))
        if y_grid_spacing is not None:  # Explicit None check
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(y_grid_spacing))

        ax.grid(True, axis='x', color=GREY91, linestyle='-', linewidth=grid_linewidth)
        ax.grid(True, axis='y', color=GREY91, linestyle='-', linewidth=grid_linewidth)
    
        ax.xaxis.set_label_position('top')
        
        if y_lim:
            ax.set_ylim(y_lim)
        
        if x_lim:
            ax.set_xlim(x_lim)
    
        ax.set_xlabel('Time (Ma)',
                    labelpad = 8.0,
                    fontname= "Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )
        
        if y_variable == 'depth':
            y_label = 'Depth (m)'
        elif y_variable == 'temp':
            y_label = 'Temperature (ºC)'
        
        ax.set_ylabel(y_label,
                    fontname= "Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )
    
        ax.tick_params(axis="x", length=2, color=GREY91)
        ax.tick_params(axis="y", length=2, color=GREY91)
    
        plt.xticks(fontsize = 10, fontname = 'Arial', color = GREY40)
        plt.yticks(fontsize = 10, fontname = 'Arial', color = GREY40)
        
        ### Final Formatting -----------------------------
        title = '{} {}'.format(self.sample_name, plot_type.capitalize())
        if plotAgeHistogram:
            fig.suptitle(title,
                      fontsize = 15,
                      fontname = "Arial",
                      color = GREY10
                      )
        else:
            plt.title(title,
                      fontsize = 15,
                      fontname = "Arial",
                      color = GREY10,
                      pad = 15
                     )
    
        plt.subplots_adjust(hspace=0)   
    
        ### Show and Save Figure -----------------------------
        if saveFig:
            pathlib.Path(saveFolder).mkdir(parents=True, exist_ok=True)
            if savefigFileName:
                filepath = '{}/{}.pdf'.format(saveFolder, savefigFileName)
            else:
                filepath = '{}/{}_{}_HeFty_plots.pdf'.format(saveFolder, self.sample_name, plot_type)
            plt.savefig(filepath, dpi = 'figure', bbox_inches='tight', pad_inches = 0.5)
    
        plt.show();

    def plot_measured_vs_modeled(self,
        measured_sample_ages: List[float],
        pathsToPlot: str = 'all',
        color_palette: str = 'Dark2',
        ahe_x_bounds: Optional[Tuple[float, float]] = None, 
        zhe_x_bounds: Optional[Tuple[float, float]] = None, 
        aft_x_bounds: Optional[Tuple[float, float]] = None, 
        zft_x_bounds: Optional[Tuple[float, float]] = None,
        ahe_y_bounds: Optional[Tuple[float, float]] = None, 
        zhe_y_bounds: Optional[Tuple[float, float]] = None, 
        aft_y_bounds: Optional[Tuple[float, float]] = None, 
        zft_y_bounds: Optional[Tuple[float, float]] = None,
        show_1v1_line: Union[bool, str] = True,
        saveFig: bool = False, 
        saveFolder: str = 'Plots', 
        savefigFileName: Optional[str] = None
        ) -> None:
        """
        Creates scatter plots comparing measured ages against modeled ages for different thermochronometer types within a single sample model.
        Automatically organizes data by thermochronometer type and creates separate subplots for each type present
        in the dataset.

        Parameters
        ----------
        measured_sample_ages : list[float]
            List of measured ages corresponding to samples in self.sample_grain_list.
            Must be in the same order as the samples in sample_grain_list.

        pathsToPlot : str, default='all'
            Controls which thermal history paths to include in the visualization. Options are:
            - 'good': Shows only good-fit paths (including best path)
            - 'acc': Shows only acceptable-fit paths
            - 'all': Shows both good and acceptable paths
        
        color_palette : str, default='Dark2'
            Name of the matplotlib colormap to use for the plot. 

        ahe_x_bounds : tuple[float, float], optional
            Custom x-axis limits for apatite plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zhe_x_bounds : tuple[float, float], optional
            Custom x-axis limits for zircon plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        aft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for AFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for ZFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        ahe_y_bounds : tuple[float, float], optional
            Custom y-axis limits for apatite plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zhe_y_bounds : tuple[float, float], optional
            Custom y-axis limits for zircon plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        aft_y_bounds : tuple[float, float], optional
            Custom y-axis limits for AFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zft_y_bounds : tuple[float, float], optional
            Custom y-axis limits for ZFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        show_1v1_line : Union[bool, str], default=True
            Controls the display of the 1:1 line and/or points. Options are:
            - True or 'both': Shows both line and points
            - 'line': Shows only the 1:1 line
            - 'point': Shows only the points
            - False: Shows neither

        saveFig : bool, default=False
            If True, saves the figure to disk.

        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            Will be created if it doesn't exist.

        savefigFileName : str, optional
            Custom filename for the saved figure. If None and saveFig is True,
            a default filename will be generated based on the plot type.

        Returns
        -------
        None
            Function displays the plot and optionally saves it to disk.

        Raises
        -------
            ValueError: If the length of measured_sample_ages does not match self.sample_grain_list.

        Notes
        -----
        - The function creates separate subplots for each thermochronometer type present in the data (AHe, ZHe, AFT, ZFT).
        - Acceptable paths are plotted with lighter shades of the same colors used for good paths.
        - Best-fit paths are always included with the good-fit paths.

        """
        if len(measured_sample_ages) != len(self.sample_grain_list):
            raise ValueError("The number of measured ages must match the number of samples in sample_grain_list.")
        
        # Filter paths based on pathsToPlot parameter
        filtered_path_dict = {}
        for path_id, path_data in self.path_dict.items():
            if 'best' in path_id.lower():
                filtered_path_dict[path_id] = path_data
            elif 'good' in path_id.lower() and pathsToPlot in ['all', 'good']:
                filtered_path_dict[path_id] = path_data
            elif 'acc' in path_id.lower() and pathsToPlot in ['all', 'acc']:
                filtered_path_dict[path_id] = path_data

        # Store original path_dict and replace with filtered version
        original_path_dict = self.path_dict
        self.path_dict = filtered_path_dict

        try:
            # Separate samples by type
            sample_types = {
                "apatite": [],
                "zircon": [],
                "aft": [],
                "zft": []
            }
            for sample, measured_age in zip(self.sample_grain_list, measured_sample_ages):
                sample_lower = sample.lower()
                for sample_type in sample_types:
                    if sample_type in sample_lower:
                        sample_types[sample_type].append((sample, measured_age))

            # Remove sample types with no data
            sample_types = {k: v for k, v in sample_types.items() if v}

            # Create subplots dynamically based on the number of sample types
            num_sample_types = len(sample_types)
            fig, axes = plt.subplots(1, num_sample_types, figsize=(6 * num_sample_types, 6), sharey=False)
            if num_sample_types == 1:
                axes = [axes]  # Ensure axes is iterable for a single subplot

            # Helper function to generate a lighter version of a color
            def get_lighter_color(color, factor=0.6):
                """Convert color to lighter version by mixing with white"""
                if isinstance(color, str):
                    color = matplotlib.colors.to_rgb(color)
                return tuple(c + (1 - c) * factor for c in color)

            # Bounds for each sample type
            x_bounds_dict = {
                "apatite": ahe_x_bounds,
                "zircon": zhe_x_bounds,
                "aft": aft_x_bounds,
                "zft": zft_x_bounds
            }
            y_bounds_dict = {
                "apatite": ahe_y_bounds,
                "zircon": zhe_y_bounds,
                "aft": aft_y_bounds,
                "zft": zft_y_bounds
            }

            def plot_sample_type(ax, sample_type, sample_data, x_bounds, y_bounds, show_1v1_line):
                """Helper function to plot measured vs. modeled ages for a specific sample type."""
                # Get color palette
                num_samples = len(sample_data)
                color_cycle = plt.cm.get_cmap(color_palette)
                
                for idx, (sample, measured_age) in enumerate(sample_data):
                    # Get base color for this sample
                    base_color = color_cycle(idx / plt.cm.get_cmap(color_palette).N)
                    lighter_color = get_lighter_color(base_color)
                    
                    # Collect modeled ages based on path type
                    good_ages = []
                    acc_ages = []
                    best_age = None
                    
                    for path_id, path_data in self.path_dict.items():
                        if sample in path_data and 'modeled_age' in path_data[sample]:
                            if 'best' in path_id.lower():
                                best_age = path_data[sample]['modeled_age']
                            elif 'good' in path_id.lower():
                                good_ages.append(path_data[sample]['modeled_age'])
                            elif 'acc' in path_id.lower():
                                acc_ages.append(path_data[sample]['modeled_age'])

                    # Plot acceptable paths first (if included)
                    if acc_ages:
                        ax.scatter(
                            [measured_age] * len(acc_ages),
                            acc_ages,
                            color=lighter_color,
                            s=80,
                            alpha=0.7
                        )

                    # Plot good paths and best path together
                    if good_ages or best_age is not None:
                        # Combine good ages with best age if it exists
                        all_good_ages = good_ages
                        if best_age is not None:
                            all_good_ages.append(best_age)
                        
                        ax.scatter(
                            [measured_age] * len(all_good_ages),
                            all_good_ages,
                            color=base_color,
                            label=f"{sample}",
                            s=80,
                            alpha=0.7
                        )

                # Add measured age point or line if requested
                if show_1v1_line in ['both', 'line']:
                    ax.axline((0, 0), slope=1, color='gray', linestyle='--', label="1:1 Line")

                if show_1v1_line in ['both', 'point']:
                    measured_ages = [data[1] for data in sample_data]
                    ax.scatter(measured_ages, measured_ages, c='k', s=80, alpha=1.0, label="1:1 Age")

                ax.set_xlabel("Measured Age (Ma)")
                ax.set_ylabel("Modeled Age (Ma)")

                if sample_type in ['aft', 'zft']:
                    title = f"Measured vs. Modeled Age for {sample_type.upper()} Samples - {self.sample_name}"
                else:
                    title = f"Measured vs. Modeled Age for {sample_type.capitalize()} Samples - {self.sample_name}"
                ax.set_title(title)

                if x_bounds:
                    ax.set_xlim(x_bounds)
                if y_bounds:
                    ax.set_ylim(y_bounds)

                # Customize legend with only one entry per sample
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), 
                        title="Samples", fontsize=10, 
                        loc='best')
                ax.grid(True)

            # Plot each sample type
            for ax, (sample_type, sample_data) in zip(axes, sample_types.items()):
                x_bounds = x_bounds_dict[sample_type]
                y_bounds = y_bounds_dict[sample_type]
                plot_sample_type(ax, sample_type, sample_data, x_bounds, y_bounds, show_1v1_line)

            # Adjust layout
            plt.tight_layout()

            # Show and Save Figure
            if saveFig:
                pathlib.Path(saveFolder).mkdir(parents=True, exist_ok=True)
                if savefigFileName:
                    filepath = f'{saveFolder}/{savefigFileName}.pdf'
                else:
                    filepath = f'{saveFolder}/Measured_vs_Modeled_Ages_{self.sample_name}.pdf'
                plt.savefig(filepath, dpi='figure', bbox_inches='tight', pad_inches=0.5)

            plt.show()
            pass
        finally:
            # Restore original path_dict
            self.path_dict = original_path_dict

    def plot_modeled_age_histograms(self, 
        bin_type: str = 'count',
        bins: Union[int, float] = 20, 
        share_x: bool = False, 
        whatToPlot: str = 'both', 
        pathsToPlot: str = 'all', 
        ap_x_bounds: Optional[Tuple[float, float]] = None, 
        zr_x_bounds: Optional[Tuple[float, float]] = None, 
        aft_x_bounds: Optional[Tuple[float, float]] = None, 
        zft_x_bounds: Optional[Tuple[float, float]] = None,
        saveFig: bool = False, 
        saveFolder: str = 'Plots', 
        savefigFileName: Optional[str] = None
        ) -> None:
        """
        Creates visualizations of modeled age distributions for different thermochronometer types within a single sample model.
        Plots can include histograms and/or kernel density estimates of modeled ages, with options for filtering path types and customizing the appearance.

        Parameters
        ----------
        bins : Union[int, float], default=20
            If bin_type is 'count': Number of bins to use in the histogram
            If bin_type is 'width': Width of each bin in Ma, starting from 0

        bin_type : str, default='count'
            Specifies how to interpret the bins parameter. Options are:
            - 'count': bins specifies the number of bins
            - 'width': bins specifies the width of each bin in Ma

        share_x : bool, default=False
            If True, all subplots will share the same x-axis limits. Useful when comparing age distributions
            across different thermochronometer types.

        whatToPlot : str, default='both'
            Specifies the type of visualization to create. Options are:
            - 'histogram': Shows only histogram representation
            - 'kde': Shows only kernel density estimate
            - 'both': Shows both histogram and KDE overlay

        pathsToPlot : str, default='all'
            Controls which thermal history paths to include in the visualization. Options are:
            - 'good': Shows only good-fit paths (including best path)
            - 'acc': Shows only acceptable-fit paths
            - 'all': Shows both good and acceptable paths

        ap_x_bounds : tuple[float, float], optional
            Custom x-axis limits for apatite plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zr_x_bounds : tuple[float, float], optional
            Custom x-axis limits for zircon plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        aft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for AFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for ZFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        saveFig : bool, default=False
            If True, saves the figure to disk.

        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            Will be created if it doesn't exist.

        savefigFileName : str, optional
            Custom filename for the saved figure. If None and saveFig is True,
            a default filename will be generated based on the plot type.

        Returns
        -------
        None
            Function displays the plot and optionally saves it.
        
        Notes
        -----
        - The function automatically detects available thermochronometer types and creates appropriate subplots.
        - Best-fit paths are always included with the good-fit paths.
        - Acceptable paths are plotted with lighter shades of the same colors used for good paths.
        """
        if not self.sample_grain_list or not self.path_dict:
            raise ValueError("No data available to plot.")
        
        if bin_type not in ['count', 'width']:
            raise ValueError("bin_type must be either 'count' or 'width'")
        
        # Filter paths based on pathsToPlot parameter
        filtered_path_dict = {}
        for path_id, path_data in self.path_dict.items():
            if 'best' in path_id.lower():
                filtered_path_dict[path_id] = path_data
            elif 'good' in path_id.lower() and pathsToPlot in ['all', 'good']:
                filtered_path_dict[path_id] = path_data
            elif 'acc' in path_id.lower() and pathsToPlot in ['all', 'acc']:
                filtered_path_dict[path_id] = path_data

        # Store original path_dict and replace with filtered version
        original_path_dict = self.path_dict
        self.path_dict = filtered_path_dict

        try:
            # Define sample types
            sample_types = ['apatite', 'zircon', 'aft', 'zft']
            available_samples = {
                sample_type: [
                    s for s in self.sample_grain_list if sample_type in s.lower()
                ]
                for sample_type in sample_types
            }
            available_sample_types = [
                sample_type for sample_type, samples in available_samples.items() if samples
            ]

            # Remove empty sample types
            sample_types = {k: v for k, v in available_samples.items() if v}

            # Define x-bounds dictionary
            x_bounds_dict = {
                'apatite': ap_x_bounds,
                'zircon': zr_x_bounds,
                'aft': aft_x_bounds,
                'zft': zft_x_bounds
            }

            if not sample_types:
                raise ValueError("No samples matching the specified types found.")

            # Set up subplots based on available sample types
            n_subplots = len(sample_types)
            fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 2.5 * n_subplots), sharex=share_x)
            if n_subplots == 1:
                axes = [axes]  # Ensure axes is iterable if only one plot

            # Helper function to generate lighter color
            def get_lighter_color(color, factor=0.6):
                """Convert color to lighter version by mixing with white"""
                if isinstance(color, str):
                    color = matplotlib.colors.to_rgb(color)
                return tuple(c + (1 - c) * factor for c in color)

            # Helper function to plot histograms for a group of samples
            def plot_group_histograms(ax, samples, sample_type, whatToPlot, x_bounds):
                # Get color palette
                colors = plt.cm.tab10.colors
                color_idx = 0

                for sample in samples:
                    modeled_ages = [
                        path_data[sample]['modeled_age']
                        for path_id, path_data in self.path_dict.items()
                        if sample in path_data and 'modeled_age' in path_data[sample]
                    ]

                    if modeled_ages:
                        # Calculate bins based on bin_type
                        if bin_type == 'width':
                            max_age = max(modeled_ages)
                            num_bins = int(np.ceil(max_age / bins))
                            bin_edges = np.arange(0, (num_bins + 1) * bins, bins)
                            hist_bins = bin_edges
                        else:  # bin_type == 'count'
                            hist_bins = bins
                        
                        current_color = colors[color_idx % len(colors)]

                        if whatToPlot in ('histogram'):
                            ax.hist(
                                modeled_ages,
                                bins=hist_bins,
                                alpha=0.6,
                                color=current_color,
                                label=sample,
                                edgecolor='black',
                                density=False
                            )
                        if whatToPlot in ('both') and len(modeled_ages) > 1:
                            ax.hist(
                                modeled_ages,
                                bins=hist_bins,
                                alpha=0.6,
                                color=current_color,
                                label=sample,
                                edgecolor='black',
                                density=True
                            )
                            
                            # Plot KDE without label
                            kde = gaussian_kde(modeled_ages)
                            x_vals = np.linspace(min(modeled_ages), max(modeled_ages), 1000)
                            y_vals = kde(x_vals)
                            ax.plot(
                                x_vals,
                                y_vals,
                                color=current_color,
                                linestyle='--',
                                label='_nolegend_'
                            )

                        if whatToPlot in ('kde') and len(modeled_ages) > 1:
                            kde = gaussian_kde(modeled_ages)
                            x_vals = np.linspace(min(modeled_ages), max(modeled_ages), 1000)
                            y_vals = kde(x_vals)
                            ax.plot(
                                x_vals,
                                y_vals,
                                color=current_color,
                                linestyle='--',
                                label=sample
                            )

                        color_idx += 1

                # Set title with sample name
                if sample_type in ('apatite', 'zircon'):
                    title = f"{sample_type.capitalize()} Modeled Age Histograms - {self.sample_name}"
                else:
                    title = f"{sample_type.upper()} Modeled Age Histograms - {self.sample_name}"
                ax.set_title(title, fontsize=14)
                
                ax.set_ylabel("Frequency" if whatToPlot == 'histogram' else "Density", fontsize=12)

                # Set x-axis bounds if provided
                if x_bounds:
                    ax.set_xlim(x_bounds)

                if samples:
                    ax.legend(title="Samples", fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)

            # Plot histograms for each sample type
            for ax, (sample_type, samples) in zip(axes, sample_types.items()):
                x_bounds = x_bounds_dict[sample_type]
                plot_group_histograms(ax, samples, sample_type, whatToPlot, x_bounds)

            # Label the x-axis on the last subplot
            axes[-1].set_xlabel("Modeled Age (Ma)", fontsize=12)

            # Adjust layout and display
            plt.tight_layout()

            # Show and Save Figure
            if saveFig:
                pathlib.Path(saveFolder).mkdir(parents=True, exist_ok=True)
                if savefigFileName:
                    filepath = f'{saveFolder}/{savefigFileName}.pdf'
                else:
                    filepath = f'{saveFolder}/Histogram_of_Modeled_Ages.pdf'
                plt.savefig(filepath, dpi='figure', bbox_inches='tight', pad_inches=0.5)

            plt.show()
            pass
        finally:
            # Restore original path_dict
            self.path_dict = original_path_dict



class MultiSampleModel:
    def __init__(self, folder_path: str, encoding: str = 'latin-1'):
        self.folder_path = folder_path
        self.encoding = encoding
        self.samples: Dict[str, Dict[str, SingleSampleModel]] = {}  # Store sample models by sample name and type
        self.master_sample: Dict[str, SingleSampleModel] = {}  # Store both 'depth' and 'temp' SingleSampleModels for the master sample
        self.best_paths: Dict[str, Dict[str, Dict]] = {}  # Store best paths organized by sample name and type
        self.organize_files()
    
    def __repr__(self):
        """
        Returns a string representation of the MultiSampleModel object.
        """
        sample_keys = list(self.samples.keys())
        if self.master_sample:
            master_sample_name = self.master_sample['sample_name']
            master_data_types = list(self.master_sample['models'].keys())
            master_info = (master_sample_name, master_data_types)
        else:
            master_info = None

        return (f"MultiSampleModel(folder_path='{self.folder_path}', "
                f"samples={sample_keys}, master_sample={master_info})")

    def organize_files(self):
        """
        Organizes files in the folder and initializes sample models.
        Determines the master sample based on file types ('depth' for master sample).
        Only processes .txt files and ignores other file types.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"The folder '{self.folder_path}' does not exist.")

        filenames = [
            file for file in os.listdir(self.folder_path)
            if os.path.isfile(os.path.join(self.folder_path, file)) and file.endswith('.txt')
        ]
        
        if not filenames:
            raise ValueError(f"No .txt files found in '{self.folder_path}'. Make sure your files have the .txt extension.")


        for file in filenames:
            # Check if the file follows the expected naming pattern
            if '-inv' not in file:
                print(f"Warning: Skipping '{file}' as it doesn't follow the expected naming pattern (should contain '-inv').")
                continue
                
            try:
                sample, suffix = file.split('-inv')
                file_type = 'depth' if 'tZ' in suffix else 'temp'

                # Initialize a SingleSampleModel for the file
                sample_model = SingleSampleModel(file_name=os.path.join(self.folder_path, file), sample_name = sample, encoding=self.encoding)

                # Add to the samples dictionary
                if sample not in self.samples:
                    self.samples[sample] = {}
                    self.best_paths[sample] = {}

                self.samples[sample][file_type] = sample_model

                # Store the best path for this sample and file type
                if hasattr(sample_model, 'best_path'):
                    self.best_paths[sample][file_type] = sample_model.best_path

                # Determine the master sample
                if file_type == 'depth':
                    self.master_sample = {'sample_name': sample, 'models': self.samples[sample]}
            
            except Exception as e:
                print(f"Warning: Error processing file '{file}': {str(e)}")
                continue
                
        if not self.samples:
            raise ValueError(f"No valid HeFTy txt files could be processed from '{self.folder_path}'.")
        
        if not self.master_sample:
            print("Warning: No master sample with depth data was found. Multi-sample path family analysis will not be available.")

    def get_best_paths(self) -> Dict[str, Dict[str, Dict]]:
        """
        Returns the dictionary of best paths for all samples.

        Returns:
            Dict[str, Dict[str, Dict]]: A nested dictionary containing best paths organized by
                                      sample name and type (depth/temp).
        """
        return self.best_paths

    def get_sample(self, sample: str) -> Optional[Dict[str, SingleSampleModel]]:
        """
        Retrieves sample models by their sample name.

        Parameters:
            sample (str): The name of the sample.

        Returns:
            Optional[Dict[str, SingleSampleModel]]: Dictionary containing 'depth' and 'temp' SingleSampleModels.
        """
        return self.samples.get(sample)

    def get_sample_best_path(self, sample: str, path_type: str) -> Optional[Dict]:
        """
        Retrieves the best path for a specific sample and path type.

        Parameters:
            sample (str): The name of the sample
            path_type (str): The type of path ('depth' or 'temp')

        Returns:
            Optional[Dict]: The best path data for the specified sample and type,
                          or None if not found
        """
        return self.best_paths.get(sample, {}).get(path_type)

    def list_samples_and_types(self) -> List[Tuple[str, List[str]]]:
        """
        Returns a list of sample names and their associated file types.

        Returns:
            List[Tuple[str, List[str]]]: A list of tuples where each tuple contains the sample name and
                                         a list of file types (e.g., ['depth', 'temp']).
        """
        return [(prefix, list(types.keys())) for prefix, types in self.samples.items()]

    def plotMultiSampleModeledAgeHistogram(self, 
        sample: str, 
        variable: str, 
        bin_type: str = 'count', 
        bins: Union[int,float] = 20, 
        share_x: bool = False, 
        whatToPlot: str = 'both', 
        pathsToPlot: str = 'all', 
        ahe_x_bounds: Optional[Tuple[float, float]] = None, 
        zhe_x_bounds: Optional[Tuple[float, float]] = None, 
        aft_x_bounds: Optional[Tuple[float, float]] = None, 
        zft_x_bounds: Optional[Tuple[float, float]] = None,
        saveFig: bool = False, 
        saveFolder: str = 'Plots', 
        savefigFileName: Optional[str] = None
        ) -> None:
        """
        Creates visualizations of modeled age distributions for different thermochronometer types
        within a specific sample from the multi-sample model. Plots can include histograms and/or
        kernel density estimates of modeled ages, with options for filtering path types and
        customizing the appearance.

        Parameters
        ----------
        sample : str
            The name of the sample to plot from the multi-sample model.

        variable : str
            Type of data to use for the visualization. Options are:
            - 'temp': Temperature data
            - 'depth': Depth data (only available for certain samples)

        bin_type : str, default='count'
            Specifies how to interpret the bins parameter. Options are:
            - 'count': bins specifies the number of bins
            - 'width': bins specifies the width of each bin in Ma

        bins : Union[int, float], default=20
            If bin_type is 'count': Number of bins to use in the histogram
            If bin_type is 'width': Width of each bin in Ma, starting from 0

        share_x : bool, default=False
            If True, all subplots will share the same x-axis limits. Useful when comparing age distributions
            across different thermochronometer types.

        whatToPlot : str, default='both'
            Specifies the type of visualization to create. Options are:
            - 'histogram': Shows only histogram representation
            - 'kde': Shows only kernel density estimate
            - 'both': Shows both histogram and KDE overlay

        pathsToPlot : str, default='all'
            Controls which thermal history paths to include in the visualization. Options are:
            - 'good': Shows only good-fit paths (including best path)
            - 'acc': Shows only acceptable-fit paths
            - 'all': Shows both good and acceptable paths

        ahe_x_bounds : tuple[float, float], optional
            Custom x-axis limits for apatite plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zhe_x_bounds : tuple[float, float], optional
            Custom x-axis limits for zircon plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        aft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for AFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for ZFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        saveFig : bool, default=False
            If True, saves the figure to disk.

        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            Will be created if it doesn't exist.

        savefigFileName : str, optional
            Custom filename for the saved figure. If None and saveFig is True,
            a default filename will be generated based on the plot type.

        Returns
        -------
        None
            Function displays the plot and optionally saves it.
        
        Notes
        -----
        - The function automatically detects available thermochronometer types and creates appropriate subplots.
        - Best-fit paths are always included with the good-fit paths.
        - Acceptable paths are plotted with lighter shades of the same colors used for good paths.
        """
        # Check if the sample exists
        if sample not in self.samples:
            available_samples = list(self.samples.keys())
            print(f"Error: Sample '{sample}' not found. Available samples are: {available_samples}")
            return
        
        # Check if the variable type exists for this sample
        if variable not in self.samples[sample]:
            available_variables = list(self.samples[sample].keys())
            print(f"Error: Variable '{variable}' not available for sample '{sample}'.")
            print(f"Available variables for this sample are: {available_variables}")
            return

        sample_model = self.get_sample(sample)[variable]

        # Call the modeled age histogram plotting method (from SingleSampleModel)
        sample_model.plot_modeled_age_histograms(bin_type, bins, share_x, whatToPlot, pathsToPlot, 
                                                 ahe_x_bounds, zhe_x_bounds, aft_x_bounds, zft_x_bounds,
                                                 saveFig, saveFolder, savefigFileName)

    def plotMultiSampleModeled_v_Measured(self, 
        sample: str,
        variable: str,
        measured_sample_ages: List[float],
        pathsToPlot: str = 'all',
        colorPalette: str = 'Set1',
        ahe_x_bounds: Optional[Tuple[float, float]] = None,
        zhe_x_bounds: Optional[Tuple[float, float]] = None,
        aft_x_bounds: Optional[Tuple[float, float]] = None,
        zft_x_bounds: Optional[Tuple[float, float]] = None,
        ahe_y_bounds: Optional[Tuple[float, float]] = None,
        zhe_y_bounds: Optional[Tuple[float, float]] = None,
        aft_y_bounds: Optional[Tuple[float, float]] = None,
        zft_y_bounds: Optional[Tuple[float, float]] = None,
        show_1v1_line: Union[bool, str] = True,
        saveFig: bool = False,
        saveFolder: str = 'Plots',
        savefigFileName: Optional[str] = None
        ) -> None:
        """
        Creates scatter plots comparing measured ages against modeled ages for different thermochronometer types
        within a specific sample from the multi-sample model.

        Parameters
        ----------
        sample : str
            The name of the sample to plot from the multi-sample model.

        variable : str
            Type of data to use for the visualization. Options are:
            - 'temp': Temperature data
            - 'depth': Depth data (only available for master sample)

        measured_sample_ages : List[float]
            List of measured ages corresponding to samples in the sample_grain_list.
            Must be in the same order as the samples in sample_grain_list.
            Use list_sample_grains() or get_sample_grains() to see the correct order.

        pathsToPlot : str, default='all'
            Controls which thermal history paths to include in the visualization. Options are:
            - 'good': Shows only good-fit paths (including best path)
            - 'acc': Shows only acceptable-fit paths
            - 'all': Shows both good and acceptable paths
        
        colorPalette : str, default='Set1'
            Name of the matplotlib colormap to use for the plot.

        ahe_x_bounds : tuple[float, float], optional
            Custom x-axis limits for apatite plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zhe_x_bounds : tuple[float, float], optional
            Custom x-axis limits for zircon plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        aft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for AFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zft_x_bounds : tuple[float, float], optional
            Custom x-axis limits for ZFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        ahe_y_bounds : tuple[float, float], optional
            Custom y-axis limits for apatite plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zhe_y_bounds : tuple[float, float], optional
            Custom y-axis limits for zircon plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        aft_y_bounds : tuple[float, float], optional
            Custom y-axis limits for AFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        zft_y_bounds : tuple[float, float], optional
            Custom y-axis limits for ZFT plots in the form (min_age, max_age).
            If None, limits are automatically determined from the data.

        show_1v1_line : Union[bool, str], default=True
            Controls the display of the 1:1 line and/or points. Options are:
            - True or 'both': Shows both line and points
            - 'line': Shows only the 1:1 line
            - 'point': Shows only the points
            - False: Shows neither

        saveFig : bool, default=False
            If True, saves the figure to disk.

        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            Will be created if it doesn't exist.

        savefigFileName : str, optional
            Custom filename for the saved figure. If None and saveFig is True,
            a default filename will be generated based on the plot type.

        Returns
        -------
        None
            Function displays the plot and optionally saves it to disk.

        Raises
        ------
        ValueError
            - If the sample doesn't exist in the model
            - If the variable type isn't available for the sample
            - If the length of measured_sample_ages doesn't match the sample_grain_list

        Notes
        -----
        - The function creates separate subplots for each thermochronometer type present in the data (AHe, ZHe, AFT, ZFT).
        - Acceptable paths are plotted with lighter shades of the same colors used for good paths.
        - Best-fit paths are always included with the good-fit paths.
        - Use list_sample_grains() or get_sample_grains() to see the required order for measured_sample_ages.
        """
        # Check if the sample exists
        if sample not in self.samples:
            available_samples = list(self.samples.keys())
            raise ValueError(f"Sample '{sample}' not found. Available samples are: {available_samples}")
        
        # Check if the variable type exists for this sample
        if variable not in self.samples[sample]:
            available_variables = list(self.samples[sample].keys())
            raise ValueError(
                f"Variable '{variable}' not available for sample '{sample}'. "
                f"Available variables are: {available_variables}"
            )
        
        # Get the sample model
        sample_model = self.get_sample(sample)[variable]
        
        # Check if measured_sample_ages matches the length of sample_grain_list
        if len(measured_sample_ages) != len(sample_model.sample_grain_list):
            grain_list = sample_model.sample_grain_list
            raise ValueError(
                f"The number of measured ages ({len(measured_sample_ages)}) must match the number of "
                f"samples in sample_grain_list ({len(grain_list)}). The expected grain list is: {grain_list}"
            )
        
        # Call the measured vs. modeled plotting method (from SingleSampleModel)
        sample_model.plot_measured_vs_modeled(
            measured_sample_ages=measured_sample_ages,
            pathsToPlot=pathsToPlot,
            color_palette=colorPalette,
            ahe_x_bounds=ahe_x_bounds,
            zhe_x_bounds=zhe_x_bounds,
            aft_x_bounds=aft_x_bounds,
            zft_x_bounds=zft_x_bounds,
            ahe_y_bounds=ahe_y_bounds,
            zhe_y_bounds=zhe_y_bounds,
            aft_y_bounds=aft_y_bounds,
            zft_y_bounds=zft_y_bounds,
            show_1v1_line=show_1v1_line,
            saveFig=saveFig,
            saveFolder=saveFolder,
            savefigFileName=savefigFileName
        )
    
    def list_sample_grains(self, print_output: bool = True) -> Dict[str, Dict[str, List[str]]]:
        """
        Lists all grains associated with each sample in the multi-sample model.
        
        This function iterates through all samples in the multi-sample model and
        retrieves the grain lists for each variable type ('temp' and/or 'depth').
        
        Parameters
        ----------
        print_output : bool, default=True
            If True, prints a formatted table of samples, variables, and grains.
            If False, only returns the dictionary without printing.
        
        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            A nested dictionary structure where:
            - The outer keys are sample names
            - The inner keys are variable types ('temp' or 'depth')
            - The values are lists of grain names associated with each sample/variable
        
        Notes
        -----
        - This function is particularly useful when preparing input for the
        plotMultiSampleModeled_v_Measured function, which requires measured ages
        to be provided in the same order as the grains in sample_grain_list.
        """
        grain_dict = {}

        # Iterate through all samples
        for sample_name, variable_dict in self.samples.items():
            grain_dict[sample_name] = {}
            
            # Iterate through variables (temp or depth)
            for variable_type, sample_model in variable_dict.items():
                # Get the grain list from the sample model
                grain_dict[sample_name][variable_type] = sample_model.sample_grain_list
        
        # Print formatted output if requested
        if print_output:
            print("\n===== SAMPLE GRAIN BREAKDOWN =====")
            
            # Find the longest sample name for formatting
            master_sample_name = None
            if hasattr(self, 'master_sample') and self.master_sample:
                master_sample_name = self.master_sample.get('sample_name')
            
            # Adjust max length calculation to account for possible asterisk
            max_sample_len = max([len(s) + (2 if s == master_sample_name else 0) for s in self.samples.keys()])
            max_var_len = 5  # 'depth' is 5 characters
            
            # Print header
            print(f"\nSAMPLE{' ' * (max_sample_len - 6)} | TYPE  | GRAINS")
            print("-" * (max_sample_len + max_var_len + 50))
            
            # Create a sorted list of samples with master sample first
            sorted_samples = list(grain_dict.keys())
            if master_sample_name and master_sample_name in sorted_samples:
                sorted_samples.remove(master_sample_name)
                sorted_samples.insert(0, master_sample_name)
            
            # Print data rows
            for sample_name in sorted_samples:
                variable_dict = grain_dict[sample_name]
                is_master = (sample_name == master_sample_name)
                
                # Process each variable type
                for i, (variable_type, grain_list) in enumerate(sorted(variable_dict.items())):
                    # For the first variable type, print the sample name
                    if i == 0:
                        if is_master:
                            sample_display = f"{sample_name} *"
                        else:
                            sample_display = sample_name
                    else:
                        sample_display = ""
                    
                    if not grain_list:
                        grain_display = "None"
                    else:
                        grain_display = ", ".join(grain_list)
                    
                    print(f"{sample_display.ljust(max_sample_len)} | {variable_type.ljust(5)} | {grain_display}")
                
                print("-" * (max_sample_len + max_var_len + 50))
            
            # Add note about the master sample
            if master_sample_name:
                print(f"* Indicates the master sample (containing both temp and depth data)")
            
            # Return None when printing output
            return None
        
        # Return the dictionary when not printing
        return grain_dict

    def get_sample_grains(self, sample: str, variable: str = None) -> Union[List[str], Dict[str, List[str]]]:
        """
        Retrieves the grain list for a specific sample, optionally filtered by variable type.
        
        Parameters
        ----------
        sample : str
            The name of the sample to retrieve grains for.
        
        variable : str, optional
            Type of data to retrieve grains for ('temp' or 'depth').
            If None, returns grains for all available variable types for the sample.
        
        Returns
        -------
        Union[List[str], Dict[str, List[str]]]
            If variable is specified: A list of grain names for that sample and variable.
            If variable is None: A dictionary mapping variable types to lists of grain names.
        
        Raises
        ------
        ValueError
            If the specified sample doesn't exist or the requested variable 
            type is not available for the sample.
        """
        # Check if the sample exists
        if sample not in self.samples:
            available_samples = list(self.samples.keys())
            raise ValueError(f"Sample '{sample}' not found. Available samples are: {available_samples}")
        
        # If variable type is specified
        if variable is not None:
            # Check if the variable type exists for this sample
            if variable not in self.samples[sample]:
                available_variables = list(self.samples[sample].keys())
                raise ValueError(
                    f"Variable '{variable}' not available for sample '{sample}'. "
                    f"Available variables are: {available_variables}"
                )
            
            # Return the grain list for the specified sample and variable
            return self.samples[sample][variable].sample_grain_list
        
        # If no variable type is specified, return all grains for all variable types
        result = {}
        for var_type, sample_model in self.samples[sample].items():
            result[var_type] = sample_model.sample_grain_list
        
        return result
        
    def plotMultiSamplePathData(self,
        sample: str,
        plot_type: str,
        y_variable: str,
        pathsToPlot: str = 'all',
        envelopesToPlot: Optional[str] = None,
        plotOtherBestFitPaths: bool = False,
        y_lim: Optional[Tuple[float, float]] = None,
        x_lim: Optional[Tuple[float, float]] = None,
        fig_size: Optional[Tuple[float, float]] = None,
        x_grid_spacing: Optional[float] = None,  
        y_grid_spacing: Optional[float] = None,  
        grid_linewidth: float = 0.5,  
        plotAgeHistogram: bool = False,
        bin_width: int = 5,
        histogramColorPalette: str = 'Dark2',
        stat: str = 'count',
        defaultHeftyStyle: bool = True,
        bestPathColor: str = 'black',
        goodPathColor: str = 'fuchsia',
        accPathColor: str = 'limegreen',
        otherBestPathColor: str = 'dodgerblue',
        showConstraintBoxes: bool = True,
        constraintBoxColor: str = 'red',
        constraintLineColor: str = 'black',
        constraintMarkerStyle: str = 's',
        saveFig: bool = False,
        saveFolder: str = 'Plots',
        savefigFileName: Optional[str] = None
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Creates visualizations of thermal history paths for a specific sample in a multi-sample model.
        This function extends the capabilities of SingleSampleModel.plotSingleSamplePathData to work with
        multi-sample models, with special handling for master samples and inter-sample relationships.

        Parameters
        ----------
        sample : str
            Name of the sample to plot. Must match a sample name in the multi-sample model.

        plot_type : str
            Type of plot to create. Options are:
            - 'paths': Shows continuous thermal history paths
            - 'points': Shows only the constraint points (only available for master sample in depth space)
            - 'envelopes': Shows the envelope of all paths

        y_variable : str
            Variable to plot on y-axis. Options are:
            - 'temp': Temperature in °C
            - 'depth': Depth in meters

        pathsToPlot : str, default='all'
            Controls which thermal history paths to include. Options are:
            - 'good': Shows only good-fit paths (including best path)
            - 'acc': Shows only acceptable-fit paths
            - 'all': Shows both good and acceptable paths

        envelopesToPlot : str, optional
            When plot_type is 'envelopes', specifies which envelopes to show:
            - 'good': Shows only good-fit envelope
            - 'acc': Shows only acceptable-fit envelope
            - 'both': Shows both envelopes

        plotOtherBestFitPaths : bool, default=False
            If True and y_variable is 'temp', the best-fit paths from other samples 
            are plotted in dodgerblue color. Does nothing when y_variable is 'depth'.

        y_lim : tuple[float, float], optional
            Custom y-axis limits.
            If None, limits are automatically determined from the data.

        x_lim : tuple[float, float], optional
            Custom x-axis limits.
            If None, limits are automatically determined from the data.
        
        fig_size : tuple[float, float], optional
            Custom dimensions for the figure in inches as (width, height).
            If None when plotAgeHistogram is False: defaults to (16,6)
            If None when plotAgeHistogram is True: defaults to (15,8)

        x_grid_spacing : float, optional
            Spacing between vertical grid lines in Ma.
            If None, uses matplotlib's automatic grid spacing.

        y_grid_spacing : float, optional  
            Spacing between horizontal grid lines in °C or m.
            If None, uses matplotlib's automatic grid spacing.

        grid_linewidth : float, default=0.5
            Width of the grid lines.

        plotAgeHistogram : bool, default=False
            If True, adds a histogram of ages below the main plot.

        bin_width : int, default=5
            Width of bins for the age histogram in Ma.
            Only used when plotAgeHistogram is True.

        histogramColorPalette : str, default='Dark2'
            Name of the matplotlib colormap to use for the histogram.
            Only used when plotAgeHistogram is True.

        stat : str, default='count'
            Statistic to plot in histogram. Options are:
            - 'count': Shows frequency counts
            - 'density': Shows probability density
            - 'probability': Shows probability

        defaultHeftyStyle : bool, default=True
            If True, uses HeFTy-like styling with color gradients based on GOF values.
            If False, uses solid colors specified by bestPathColor, goodPathColor, and accPathColor.

        bestPathColor : str, default='black'
            Color for the best-fit path when defaultHeftyStyle is False.

        goodPathColor : str, default='fuchsia'
            Color for good-fit paths when defaultHeftyStyle is False.

        accPathColor : str, default='limegreen'
            Color for acceptable-fit paths when defaultHeftyStyle is False.
            
        otherBestPathColor : str, default='dodgerblue'
            Color for best-fit paths from other samples when plotOtherBestFitPaths is True.

        showConstraintBoxes : bool, default=True
            If True, displays constraint boxes and connecting lines.
            Only applies when the sample is the master sample and y_variable is 'depth'.

        constraintBoxColor : str, default='red'
            Color of the constraint box outlines.

        constraintLineColor : str, default='black'
            Color of the lines connecting constraint boxes.

        constraintMarkerStyle : str, default='s'
            Marker style for constraint box centers.
            Any valid matplotlib marker style.

        saveFig : bool, default=False
            If True, saves the figure to disk.

        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            Will be created if it doesn't exist.

        savefigFileName : str, optional
            Custom filename for the saved figure. If None and saveFig is True,
            a default filename will be generated based on the plot type.

        Returns
        -------
        None
            Function displays the plot and optionally saves it to disk.

        Notes
        -----
        - plot_type='points' is only available for the master sample in time-depth space.
        - Constraint boxes are only shown for the master sample in time-depth space.
        """
        # Validate input parameters
        if sample not in self.samples:
            raise ValueError(f"Sample '{sample}' not found in multi-sample model. Available samples: {list(self.samples.keys())}")
        
        if y_variable not in ('temp', 'depth'):
            raise ValueError("y_variable must be either 'temp' or 'depth'")
        
        if y_variable not in self.samples[sample]:
            raise ValueError(f"Sample '{sample}' does not have data for y_variable '{y_variable}'. Available types: {list(self.samples[sample].keys())}")
        
        # Get the sample model
        sample_model = self.samples[sample][y_variable]
        
        # Check if this is the master sample
        is_master_sample = (sample == self.master_sample.get('sample_name', ''))
        
        # Validate plot_type for special cases
        if plot_type == 'points' and (y_variable != 'depth' or not is_master_sample):
            if not is_master_sample:
                raise ValueError("'points' plot_type is only available for the master sample")
            if y_variable != 'depth':
                raise ValueError("'points' plot_type is only available in depth space")
                
        # Validate plotAgeHistogram for special cases
        if plotAgeHistogram and (not is_master_sample or y_variable != 'depth'):
            if not is_master_sample:
                print("Warning: Age histogram is only available for the master sample. Setting plotAgeHistogram to False.")
                plotAgeHistogram = False
            elif y_variable != 'depth':
                print("Warning: Age histogram is only available in depth space. Setting plotAgeHistogram to False.")
                plotAgeHistogram = False
        
        # Set up figure and axes 
        if plotAgeHistogram:
            if fig_size:
                fig_size = fig_size
            else: 
                fig_size = (15,8)

            fig, (ax, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, figsize=fig_size, sharex=True)
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:
            if fig_size:
                fig_size = fig_size
            else: 
                fig_size = (16,6)

            fig, ax = plt.subplots(1, 1, figsize=fig_size)
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()

        # Get data from the sample model
        path_dict = sample_model.get_path_data()
        constraints = sample_model.get_constraints()
        envelope_dict = sample_model.get_envelopes()

        # Set the type of plot (time/depth)
        if y_variable == 'depth':
            y_all = 'depth'
            y_con = 'depth_con'
        elif y_variable == 'temp':
            y_all = 'temp'
            y_con = 'temp_con'
        else:
            print('That is not a valid y_variable, please input either "temp" or "depth".')

        # Breaking up the dictionary to ensure correct layering for different path groups
        best_path = {}
        good_paths = {}
        acc_paths = {}

        for key, path in path_dict.items():
            if 'best' in key.lower():
                best_path[key] = path
            elif 'good' in key.lower() and (pathsToPlot in ['all', 'good']):
                good_paths[key] = path
            elif 'acc' in key.lower() and (pathsToPlot in ['all', 'acc']):
                acc_paths[key] = path

        # Plot the Age Histogram if True
        if plotAgeHistogram:
            constraintHistogramDict = {}

            # Loop through the path dictionary with filtering
            for key, path in path_dict.items():
                # Apply path filtering
                if not ('best' in key.lower() or 
                    ('good' in key.lower() and pathsToPlot in ['all', 'good']) or 
                    ('acc' in key.lower() and pathsToPlot in ['all', 'acc'])):
                    continue

                # Get the time_con list from the path dictionary - only get the "real" constraint points from model
                time_con_list = path.get('time_con', [])

                # Loop through the time_con list and group values by index
                for i, value in enumerate(time_con_list):
                    # Create a new key in the grouped dictionary if it doesn't exist
                    if i not in constraintHistogramDict:
                        constraintHistogramDict[i] = []

                    # Append the value to the corresponding key
                    constraintHistogramDict[i].append(value)

            # Plotting histogram - generate colors for all constraint keys
            cmap1 = plt.cm.get_cmap(histogramColorPalette, len(constraintHistogramDict))
            
            # set the bins
            if x_lim:
                num_bins = min(500, int((x_lim[0] - x_lim[1]) / bin_width) + 1)
                bins = np.linspace(x_lim[1], x_lim[0], num_bins)
            else:
                # Find reasonable x-limits from the data
                min_time = float('inf')
                max_time = 0
                for values in constraintHistogramDict.values():
                    if values and len(values) > 0:  # Check if list is not empty
                        min_time = min(min_time, min(values))
                        max_time = max(max_time, max(values))
                
                if min_time == float('inf') or min_time >= max_time:
                    min_time = 0
                    max_time = 100  # Default range if no valid data

                # Limit number of bins to avoid memory issues
                num_bins = min(500, int((max_time - min_time) / bin_width) + 1)
                bins = np.linspace(min_time, max_time, num_bins)

            # Plot histograms for each constraint
            for i, (constraint_index, values) in enumerate(sorted(constraintHistogramDict.items(), reverse=True)):
                if i == 0:
                    label = 'Present Day Conditions'
                elif i == -1:
                    label = f'Constraint Box {len(constraintHistogramDict) }'
                else:
                    label = f'Constraint Box {abs(i) }'
                
                # catch for if a constraint is all 0's e.g. final/present day conditions
                if min(values) == max(values) == 0:
                    pass
                else:
                    sns.histplot(ax=ax1, x=values, stat=stat, bins=bins, kde=True, color=cmap1(i), label=label)
            
            # Adding the legend
            ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)

            # Place y-axis labels on the right side
            ax1.yaxis.set_label_position("right")
            ax1.yaxis.tick_right()
            
            # Axis Label and Settings
            ax1.set_ylabel(stat.title(),
                labelpad=10,
                fontname="Arial",
                fontsize=12,
                weight=500,
                color=GREY40,
                rotation=270
            )
            
            ax1.tick_params(axis='y', length=2, color=GREY91)

        # Plot the Path Data
        # Greens colormaps to match HeFTy style
        cmap = LinearSegmentedColormap.from_list('custom cmap', [(0, 'darkolivegreen'), (1, 'lime')])

        if plot_type == 'paths':
            if pathsToPlot in ['all', 'acc']:
                for key, path in acc_paths.items():
                    color = cmap(path['comp_GOF']) if defaultHeftyStyle else accPathColor
                    ax.plot(path['time'], path[y_all], color=color, alpha=1)
            
            # Plot good paths if included
            if pathsToPlot in ['all', 'good']:
                for key, path in good_paths.items():
                    color = 'fuchsia' if defaultHeftyStyle else goodPathColor
                    ax.plot(path['time'], path[y_all], color=color, alpha=1)
            
            # Plot the color bar for GOF - default style only
            if defaultHeftyStyle and pathsToPlot in ['acc', 'all']:
                if plotAgeHistogram:  
                    # need to specify placement when histogram is turned on
                    cbar_ax = fig.add_axes([0.97, 0.3, 0.015, 0.55])  # Position of the color bar
                    
                    norm = plt.Normalize(0, 0.5)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
                    
                    cbar.ax.tick_params(axis='y', labelsize=10)
                    
                    cbar.set_label('Acc GOF Value', 
                                labelpad=15.0, 
                                fontname="Arial", 
                                fontsize=12, 
                                weight=500,
                                color=GREY40,
                                rotation=270
                                )
                else:
                    norm = plt.Normalize(0, 0.5)
                    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

                    cbar.ax.tick_params(axis='y', labelsize=10, labelcolor=GREY40)
                    for label in cbar.ax.yaxis.get_ticklabels():
                        label.set_fontname('Arial')

                    cbar.set_label('Acc GOF Value', 
                                labelpad=15.0,
                                fontname="Arial",
                                fontsize=12,
                                weight=500,
                                color=GREY40,
                                rotation=270
                                )

        elif plot_type == 'points':
            # Acceptable Paths
            for key, path in acc_paths.items():
                color = cmap(path['comp_GOF']) if defaultHeftyStyle else accPathColor
                ax.scatter(path['time_con'], path[y_con], color=color, alpha=1, marker='s')

            # Good Paths
            for key, path in good_paths.items():
                color = 'fuchsia' if defaultHeftyStyle else goodPathColor
                ax.scatter(path['time_con'], path[y_con], color=color, alpha=1, marker='s')
                
            # Plot the color bar for GOF - default style only
            if defaultHeftyStyle and pathsToPlot in ['acc', 'all']:
                if plotAgeHistogram:
                    cbar_ax = fig.add_axes([0.97, 0.3, 0.015, 0.55])  # Position of the color bar
                    
                    norm = plt.Normalize(0, 0.5)
                    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
                    
                    cbar.ax.tick_params(axis='y', labelsize=10)
                    
                    cbar.set_label('Acc GOF Value', 
                                labelpad=15.0, 
                                fontname="Arial", 
                                fontsize=12, 
                                weight=500,
                                color=GREY40,
                                rotation=270
                                )
                else:
                    norm = plt.Normalize(0, 0.5)
                    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

                    cbar.ax.tick_params(axis='y', labelsize=10, labelcolor=GREY40)
                    for label in cbar.ax.yaxis.get_ticklabels():
                        label.set_fontname('Arial')

                    cbar.set_label('Acc GOF Value', 
                                labelpad=15.0,
                                fontname="Arial",
                                fontsize=12,
                                weight=500,
                                color=GREY40,
                                rotation=270
                                )

        elif plot_type == 'envelopes':
            # Setting envelope colors based on user input 
            if defaultHeftyStyle:
                goodEnvelopeColor = 'fuchsia'
                accEnvelopeColor = 'lime'
            else:
                goodEnvelopeColor = goodPathColor
                accEnvelopeColor = accPathColor

            if y_variable == 'temp':
                if envelopesToPlot == 'both':
                    # Check to see if there are only acceptable paths
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi temp (C)'], 
                                    envelope_dict['Acc lo temp (C)'], alpha=0.7, color=accEnvelopeColor)

                        print('Note that this sample only has acceptable paths.')
                    else:
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi temp (C)'], 
                                    envelope_dict['Acc lo temp (C)'], alpha=0.7, color=accEnvelopeColor)

                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi temp (C)'], 
                                        envelope_dict['Good lo temp (C)'], alpha=0.9, color=goodEnvelopeColor)

                elif envelopesToPlot == 'acc':
                    # Acc Paths
                    ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi temp (C)'], 
                                    envelope_dict['Acc lo temp (C)'], alpha=0.7, color=accEnvelopeColor)

                elif envelopesToPlot == 'good':
                    # Verify that there are good paths 
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        print('There are no good paths for this sample. Try plotting the acceptable paths')
                    else:
                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi temp (C)'], 
                                        envelope_dict['Good lo temp (C)'], alpha=0.9, color=goodEnvelopeColor)
                        
            elif y_variable == 'depth':
                if envelopesToPlot == 'both':
                    # Check to see if there are only acceptable paths
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi depth (m)'], 
                                    envelope_dict['Acc lo depth (m)'], alpha=0.7, color=accEnvelopeColor)

                        print('Note that this sample only has acceptable paths.')
                    else:
                        # Acc Paths
                        ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi depth (m)'], 
                                    envelope_dict['Acc lo depth (m)'], alpha=0.7, color=accEnvelopeColor)

                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi depth (m)'], 
                                        envelope_dict['Good lo depth (m)'], alpha=0.9, color=goodEnvelopeColor)

                elif envelopesToPlot == 'acc':
                    # Acc Paths
                    ax.fill_between(envelope_dict['Acc time (Ma)'], envelope_dict['Acc hi depth (m)'], 
                                    envelope_dict['Acc lo depth (m)'], alpha=0.7, color=accEnvelopeColor)

                elif envelopesToPlot == 'good':
                    # Verify that there are good paths 
                    if 'acc' in list(envelope_dict.keys())[0].lower():
                        print('There are no good paths for this sample. Try plotting the acceptable paths')
                    else:
                        # Good Paths
                        ax.fill_between(envelope_dict['Good time (Ma)'], envelope_dict['Good hi depth (m)'], 
                                        envelope_dict['Good lo depth (m)'], alpha=0.9, color=goodEnvelopeColor)
                
        # Plot the best fit path         
        for key, path in best_path.items():
            path_color = 'black' if defaultHeftyStyle else bestPathColor
            ax.plot(path['time'], path[y_all], color=path_color, linewidth=3)
        
        # Add other best fit paths if requested and in temperature space
        if plotOtherBestFitPaths and y_variable == 'temp':
            for other_sample, sample_data in self.samples.items():
                # Skip the current sample
                if other_sample == sample:
                    continue
                
                # Check if this sample has temperature data and a best path
                if 'temp' in sample_data and other_sample in self.best_paths and 'temp' in self.best_paths[other_sample]:
                    best_path = self.best_paths[other_sample]['temp']
                    # Plot the other sample's best path
                    ax.plot(
                        best_path['time'], 
                        best_path['temp'], 
                        color=otherBestPathColor, 
                        alpha=0.8, 
                        linewidth=2,
                        label=f"{other_sample} best-fit"
                    )

        # Adjust constraint box display based on master sample and y_variable
        show_constraints = showConstraintBoxes and is_master_sample and y_variable == 'depth'

        if show_constraints:
            x_midpts = []
            y_midpts = []
            
            for constraint in constraints:
                max_t = constraint[0]
                min_t = constraint[1]
                max_T = constraint[2]
                min_T = constraint[3]
                
                mid_x = constraint[0] - ((constraint[0]-constraint[1])/2)
                mid_y = constraint[2] - ((constraint[2]-constraint[3])/2)
        
                x_midpts.append(mid_x)
                y_midpts.append(mid_y)
                
                height = -(max_T - min_T)
                width = -(max_t - min_t)
                
                ax.add_patch(Rectangle((max_t,max_T),width,height,
                                        edgecolor = constraintBoxColor,
                                        facecolor = None,
                                        lw = 1.5,
                                        fill = False,
                                        zorder = 100000)) #zorder - arbitrarily large number to bring to front
                
            ax.plot(x_midpts,y_midpts, linestyle = '--', color = constraintLineColor)
            ax.scatter(x_midpts,y_midpts, marker = constraintMarkerStyle, color = 'black', s = 75, facecolors = 'none', zorder = 1000)


        # Axes and Spine Customization
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

        ax.spines["left"].set_color('k')
        ax.spines["bottom"].set_color('k')
        ax.spines["right"].set_color('k')
        ax.spines["top"].set_color('k')

        if x_grid_spacing is not None: # Explicit None check
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_grid_spacing))
        if y_grid_spacing is not None: # Explicit None check
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator (y_grid_spacing))

        ax.grid(True, axis = 'x', color = GREY91, linestyle = '-', linewidth = grid_linewidth)
        ax.grid(True, axis = 'y', color = GREY91, linestyle = '-', linewidth = grid_linewidth)

        ax.xaxis.set_label_position('top')
        
        if y_lim:
            ax.set_ylim(y_lim)

        if x_lim:
            ax.set_xlim(x_lim)

        ax.set_xlabel('Time (Ma)',
                    labelpad = 8.0,
                    fontname= "Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )
        
        if y_variable == 'depth':
            y_label = 'Depth (m)'
        elif y_variable == 'temp':
            y_label = 'Temperature (ºC)'
        
        ax.set_ylabel(y_label,
                    fontname= "Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )

        ax.tick_params(axis="x", length=2, color=GREY91)
        ax.tick_params(axis="y", length=2, color=GREY91)

        plt.xticks(fontsize = 10, fontname = 'Arial', color = GREY40)
        plt.yticks(fontsize = 10, fontname = 'Arial', color = GREY40)
        
        # Update title to include multi-sample context
        title = f"{sample} {plot_type.capitalize()} - {y_variable.capitalize()}"
        if plotAgeHistogram:
            fig.suptitle(title, fontsize=15, fontname="Arial", color=GREY10)
        else:
            ax.set_title(title, fontsize=15, fontname="Arial", color=GREY10, pad=15)
        
        # Adjust layout if using histogram
        if plotAgeHistogram:
            plt.subplots_adjust(hspace=0)
        
        # Show and Save figure 
        if saveFig:
            pathlib.Path(saveFolder).mkdir(parents=True, exist_ok=True)
            if savefigFileName:
                filepath = f'{saveFolder}/{savefigFileName}.pdf'
            else:
                filepath = f'{saveFolder}/{sample}_{plot_type}_{y_variable}_MultiSample.pdf'
            plt.savefig(filepath, dpi='figure', bbox_inches='tight', pad_inches=0.5)
        
        plt.show();

    def identifyMultiSamplePathFamilies(self,
        sample_name: str,
        plot_type: str,
        y_variable: str,
        y_lim: Optional[Tuple[float, float]] = None,
        x_lim: Optional[Tuple[float, float]] = None,
        fig_size: Optional[Tuple[float, float]] = None,
        x_grid_spacing: Optional[float] = None,  
        y_grid_spacing: Optional[float] = None,  
        grid_linewidth: float = 0.5,  
        good_match_color: str = 'goldenrod',
        acc_match_color: str = 'gold',
        otherBestPathColor: str = 'dodgerblue',
        showOtherBestPaths: bool = False,
        showConstraintBoxes: bool = True,
        constraintBoxColor: str = 'red',
        constraintLineColor: str = 'black',
        constraintMarkerStyle: str = 's',
        saveFig: bool = False,
        saveFolder: str = 'Plots',
        savefigFileName: Optional[str] = None,
        c1_x: Optional[Tuple[float, float]] = None,
        c1_y: Optional[Tuple[float, float]] = None,
        c2_x: Optional[Tuple[float, float]] = None,
        c2_y: Optional[Tuple[float, float]] = None,
        c3_x: Optional[Tuple[float, float]] = None,
        c3_y: Optional[Tuple[float, float]] = None,
        c4_x: Optional[Tuple[float, float]] = None,
        c4_y: Optional[Tuple[float, float]] = None,
        c5_x: Optional[Tuple[float, float]] = None,
        c5_y: Optional[Tuple[float, float]] = None,
        c6_x: Optional[Tuple[float, float]] = None,
        c6_y: Optional[Tuple[float, float]] = None,
        c7_x: Optional[Tuple[float, float]] = None,
        c7_y: Optional[Tuple[float, float]] = None,
        c8_x: Optional[Tuple[float, float]] = None,
        c8_y: Optional[Tuple[float, float]] = None,
        c9_x: Optional[Tuple[float, float]] = None,
        c9_y: Optional[Tuple[float, float]] = None,
        c10_x: Optional[Tuple[float, float]] = None,
        c10_y: Optional[Tuple[float, float]] = None,
        c11_x: Optional[Tuple[float, float]] = None,
        c11_y: Optional[Tuple[float, float]] = None,
        c12_x: Optional[Tuple[float, float]] = None,
        c12_y: Optional[Tuple[float, float]] = None,
        c13_x: Optional[Tuple[float, float]] = None,
        c13_y: Optional[Tuple[float, float]] = None,
        c14_x: Optional[Tuple[float, float]] = None,
        c14_y: Optional[Tuple[float, float]] = None,
        c15_x: Optional[Tuple[float, float]] = None,
        c15_y: Optional[Tuple[float, float]] = None
        ) -> List[str]:
        """
        Identifies path families in multi-sample models and displays them for a specified sample.
        
        This function uses constraints applied to the master sample in time-depth space to identify
        path families, then visualizes these paths for any sample in the multi-sample model in
        either depth or temperature space.
        
        Parameters
        ----------
        sample_name : str
            Name of the sample to visualize. Must exist in the multi-sample model.
            
        plot_type : str
            Type of visualization to create. Options are:
            - 'line': Shows continuous thermal history paths
            - 'points': Shows only the constraint points (only valid for master sample in depth space)
            
        y_variable : str
            Variable to plot on y-axis. Options are:
            - 'temp': Temperature in °C
            - 'depth': Depth in meters (only valid for master sample)
            
        y_lim : tuple[float, float], optional
            Custom y-axis limits in the form (higher bound, lower bound).
            
        x_lim : tuple[float, float], optional
            Custom x-axis limits in the form (higher bound, lower bound).
            
        fig_size : tuple[float, float], optional
            Custom dimensions for the figure in inches as (width, height).
            
        x_grid_spacing : float, optional
            Spacing between vertical grid lines in Ma.
            
        y_grid_spacing : float, optional  
            Spacing between horizontal grid lines in °C or m.
            
        grid_linewidth : float, default=0.5
            Width of the grid lines.
            
        good_match_color : str, default='goldenrod'
            Color for good-fit paths that satisfy all constraints.
            
        acc_match_color : str, default='gold'
            Color for acceptable-fit paths that satisfy all constraints.
            
        otherBestPathColor : str, default='dodgerblue'
            Color for best-fit paths from other samples when showOtherBestPaths is True.
            
        showOtherBestPaths : bool, default=False
            If True, shows best-fit paths from other samples when in temperature space.
            
        showConstraintBoxes : bool, default=True
            If True, displays constraint boxes and connecting lines.
            Only applies when visualizing the master sample in depth space.
            
        constraintBoxColor : str, default='red'
            Color of the constraint box outlines.
            
        constraintLineColor : str, default='black'
            Color of the lines connecting constraint boxes.
            
        constraintMarkerStyle : str, default='s'
            Marker style for constraint box centers.
            
        saveFig : bool, default=False
            If True, saves the figure to disk.
            
        saveFolder : str, default='Plots'
            Directory where the figure should be saved if saveFig is True.
            
        savefigFileName : str, optional
            Custom filename for the saved figure.
            
        c1_x through c15_x : tuple[float, float], optional
            X-axis (time) bounds for constraints 1-15 in the form (max_time, min_time).
            These constraints are applied to the master sample in depth space.
            
        c1_y through c15_y : tuple[float, float], optional
            Y-axis (depth) bounds for constraints 1-15 in the form (max_value, min_value).
            These constraints are applied to the master sample in depth space.
            
        Returns
        -------
        List[str]
            List of path identifiers that satisfy all specified constraints.
            
        Notes
        -----
        - The function supports up to 15 user-defined constraints
        - Constraints are always applied to the master sample in depth space
        - When visualizing non-master samples, only the temperature space is available
        - The 'points' plot type is only available for the master sample in depth space
        """
        # Verify that master sample exists
        if not self.master_sample:
            raise ValueError("No master sample found in the multi-sample model.")
        
        master_sample_name = self.master_sample['sample_name']
        
        # Check if master sample has depth data
        if 'depth' not in self.master_sample['models']:
            raise ValueError(f"Master sample '{master_sample_name}' lacks depth data required for constraint identification.")
        
        # Verify the specified sample exists
        if sample_name not in self.samples:
            raise ValueError(f"Sample '{sample_name}' not found in multi-sample model. Available samples: {list(self.samples.keys())}")
        
        # Check variable constraints
        is_master = (sample_name == master_sample_name)
        
        # Depth variable can only be used with master sample
        if y_variable == 'depth' and not is_master:
            raise ValueError(f"'depth' variable can only be used with the master sample '{master_sample_name}'")
        
        # Points plot type can only be used with master sample in depth space
        if plot_type == 'points' and (not is_master or y_variable != 'depth'):
            if not is_master:
                raise ValueError(f"'points' plot type is only available for the master sample '{master_sample_name}'")
            else:
                raise ValueError("'points' plot type is only available in depth space")
        
        # Check that the requested variable exists for the sample
        if y_variable not in self.samples[sample_name]:
            raise ValueError(f"Sample '{sample_name}' does not have {y_variable} data")
        
        # Get the master sample's depth model for constraint identification
        master_sample_depth_model = self.master_sample['models']['depth']
        
        # Identify path families in master sample based on constraints
        key_matched_paths = self._identifyPathFamiliesInMaster(
            master_sample_depth_model=master_sample_depth_model,
            constraints_x=[c1_x, c2_x, c3_x, c4_x, c5_x, c6_x, c7_x, c8_x, c9_x, c10_x, c11_x, c12_x, c13_x, c14_x, c15_x],
            constraints_y=[c1_y, c2_y, c3_y, c4_y, c5_y, c6_y, c7_y, c8_y, c9_y, c10_y, c11_y, c12_y, c13_y, c14_y, c15_y]
        )
        
        # Visualization Setup
        if is_master and y_variable == 'depth' and showConstraintBoxes:
            constraints = master_sample_depth_model.get_constraints()
        else:
            constraints = None
        
        # Visualize the requested sample with the matched paths
        self._visualizeMatchedPaths(
            sample_name=sample_name,
            plot_type=plot_type,
            y_variable=y_variable,
            matched_paths=key_matched_paths,
            x_lim=x_lim,
            y_lim=y_lim,
            fig_size=fig_size,
            x_grid_spacing=x_grid_spacing,
            y_grid_spacing=y_grid_spacing,
            grid_linewidth=grid_linewidth,
            good_match_color=good_match_color,
            acc_match_color=acc_match_color,
            showOtherBestPaths=showOtherBestPaths,
            otherBestPathColor=otherBestPathColor,
            constraints=constraints,
            showConstraintBoxes=showConstraintBoxes and is_master and y_variable == 'depth',
            constraintBoxColor=constraintBoxColor,
            constraintLineColor=constraintLineColor,
            constraintMarkerStyle=constraintMarkerStyle,
            saveFig=saveFig,
            saveFolder=saveFolder,
            savefigFileName=savefigFileName
        )
        
        return key_matched_paths

    def _identifyPathFamiliesInMaster(self, 
        master_sample_depth_model,
        constraints_x: List[Optional[Tuple[float, float]]],
        constraints_y: List[Optional[Tuple[float, float]]]
        ) -> List[str]:
        """
        Helper method to identify path families in the master sample based on constraints.
        
        Parameters
        ----------
        master_sample_depth_model : SingleSampleModel
            The master sample model in depth space
            
        constraints_x : List[Optional[Tuple[float, float]]]
            List of x-axis constraints
            
        constraints_y : List[Optional[Tuple[float, float]]]
            List of y-axis constraints
            
        Returns
        -------
        List[str]
            List of path identifiers that satisfy all constraints
        """
        # Get path data
        path_dict = master_sample_depth_model.get_path_data()
        
        # Breaking up the dictionary by path types
        best_path = {}
        good_paths = {}
        acc_paths = {}
        
        for key, path in path_dict.items():
            if 'best' in key.lower():
                best_path[key] = path
            elif 'good' in key.lower():
                good_paths[key] = path
            elif 'acc' in key.lower():
                acc_paths[key] = path
        
        # Iterate over all paths and test against constraints
        matched_paths = []
        
        # Helper function to check if path meets all constraints
        def path_meets_constraints(path):
            for i, (constraint_x, constraint_y) in enumerate(zip(constraints_x, constraints_y), start=1):
                # Skip if no constraint is provided for this position
                if not constraint_x and not constraint_y:
                    continue
                    
                # Check x constraint if provided
                x_condition_met = not constraint_x or (
                    len(path['time_con']) >= (i + 1) and 
                    constraint_x[0] >= path['time_con'][-(i + 1)] >= constraint_x[1]
                )
                
                # Check y constraint if provided
                y_condition_met = not constraint_y or (
                    len(path['depth_con']) >= (i + 1) and 
                    constraint_y[0] >= path['depth_con'][-(i + 1)] >= constraint_y[1]
                )
                
                # If either constraint fails, path doesn't meet conditions
                if not (x_condition_met and y_condition_met):
                    return False
            
            # Path meets all constraints
            return True
        
        # Check all paths
        for path_collections in [best_path, good_paths, acc_paths]:
            for key, path in path_collections.items():
                if path_meets_constraints(path):
                    matched_paths.append(key)
        
        return matched_paths

    def _visualizeMatchedPaths(self,
        sample_name: str,
        plot_type: str,
        y_variable: str,
        matched_paths: List[str],
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        fig_size: Optional[Tuple[float, float]] = None,
        x_grid_spacing: Optional[float] = None,  
        y_grid_spacing: Optional[float] = None,  
        grid_linewidth: float = 0.5,
        good_match_color: str = 'goldenrod',
        acc_match_color: str = 'gold',
        showOtherBestPaths: bool = False,
        otherBestPathColor: str = 'dodgerblue',
        constraints = None,
        showConstraintBoxes: bool = False,
        constraintBoxColor: str = 'red',
        constraintLineColor: str = 'black',
        constraintMarkerStyle: str = 's',
        saveFig: bool = False,
        saveFolder: str = 'Plots',
        savefigFileName: Optional[str] = None
        ) -> None:
        """
        Helper method to visualize matched paths for a specific sample.
        
        Parameters
        ----------
        sample_name : str
            Name of the sample to visualize
            
        plot_type : str
            Type of plot ('line' or 'points')
            
        y_variable : str
            Y-axis variable ('temp' or 'depth')
            
        matched_paths : List[str]
            List of path identifiers that match constraints
            
        [Additional parameters as in identifyMultiSamplePathFamilies]
        """
        # Get sample model
        sample_model = self.samples[sample_name][y_variable]
        
        # Get path data
        path_dict = sample_model.get_path_data()
        
        # Set the y variable
        y_all = y_variable
        y_con = f"{y_variable}_con"
        
        # Prepare figure
        if fig_size:
            fig_size = fig_size
        else:
            fig_size = (12, 6)
        
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        
        # Breaking up the dictionary to ensure correct layering
        best_path = {}
        good_paths = {}
        acc_paths = {}
        
        for key, path in path_dict.items():
            if 'best' in key.lower():
                best_path[key] = path
            elif 'good' in key.lower():
                good_paths[key] = path
            elif 'acc' in key.lower():
                acc_paths[key] = path
        
        # Plot paths based on plot type
        if plot_type == 'paths':
            # First plot non-matching paths in grey
            for key, path in acc_paths.items():
                if key in matched_paths:
                    continue  # Skip matched paths for now
                path_color = 'gainsboro'
                ax.plot(path['time'], path[y_all], color=path_color, alpha=1)
            
            for key, path in good_paths.items():
                if key in matched_paths:
                    continue  # Skip matched paths for now
                path_color = 'silver'
                ax.plot(path['time'], path[y_all], color=path_color, alpha=1)
            
            # Then plot matching paths with highlight colors (to ensure they're on top)
            for key, path in acc_paths.items():
                if key in matched_paths:
                    path_color = acc_match_color
                    ax.plot(path['time'], path[y_all], color=path_color, alpha=1, zorder=10000)
            
            for key, path in good_paths.items():
                if key in matched_paths:
                    path_color = good_match_color
                    ax.plot(path['time'], path[y_all], color=path_color, alpha=1, zorder=10000)
        
        elif plot_type == 'points':
            # First plot non-matching points in grey
            for key, path in acc_paths.items():
                if key in matched_paths:
                    continue  # Skip matched paths for now
                path_color = 'gainsboro'
                ax.scatter(path['time_con'], path[y_con], color=path_color, alpha=1, marker='s')
            
            for key, path in good_paths.items():
                if key in matched_paths:
                    continue  # Skip matched paths for now
                path_color = 'silver'
                ax.scatter(path['time_con'], path[y_con], color=path_color, alpha=1, marker='s')
            
            # Then plot matching points with highlight colors
            for key, path in acc_paths.items():
                if key in matched_paths:
                    path_color = acc_match_color
                    ax.scatter(path['time_con'], path[y_con], color=path_color, alpha=1, marker='s', zorder=1000)
            
            for key, path in good_paths.items():
                if key in matched_paths:
                    path_color = good_match_color
                    ax.scatter(path['time_con'], path[y_con], color=path_color, alpha=1, marker='s', zorder=1000)
        
        # Plot the best path
        for key, path in best_path.items():
            path_color = 'black'
            ax.plot(path['time'], path[y_all], color=path_color, linewidth=3, zorder=10000)
        
        # Add other best fit paths (only in temperature space)
        if showOtherBestPaths and y_variable == 'temp':
            for other_sample, sample_data in self.samples.items():
                # Skip the current sample
                if other_sample == sample_name:
                    continue
                
                # Check if this sample has temperature data and a best path
                if 'temp' in sample_data and other_sample in self.best_paths and 'temp' in self.best_paths[other_sample]:
                    other_best_path = self.best_paths[other_sample]['temp']
                    # Plot the other sample's best path
                    ax.plot(
                        other_best_path['time'], 
                        other_best_path['temp'], 
                        color=otherBestPathColor, 
                        alpha=0.8, 
                        linewidth=2
                    )
        
        # Plot constraint boxes if available
        if showConstraintBoxes and constraints is not None:
            x_midpts = []
            y_midpts = []
            
            for constraint in constraints:
                max_t = constraint[0]
                min_t = constraint[1]
                max_T = constraint[2]
                min_T = constraint[3]
                
                mid_x = constraint[0] - ((constraint[0]-constraint[1])/2)
                mid_y = constraint[2] - ((constraint[2]-constraint[3])/2)
        
                x_midpts.append(mid_x)
                y_midpts.append(mid_y)
                
                height = -(max_T - min_T)
                width = -(max_t - min_t)
                
                ax.add_patch(Rectangle((max_t,max_T), width, height,
                                    edgecolor=constraintBoxColor,
                                    facecolor=None,
                                    lw=1.5,
                                    fill=False,
                                    zorder=100000))
            
            ax.plot(x_midpts, y_midpts, linestyle='--', color=constraintLineColor)
            ax.scatter(x_midpts, y_midpts, marker=constraintMarkerStyle, color='black', 
                    s=75, facecolors='none', zorder=1000)
        
        # Axes and Spine Customization
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        
        ax.spines["left"].set_color('k')
        ax.spines["bottom"].set_color('k')
        ax.spines["right"].set_color('k')
        ax.spines["top"].set_color('k')
        
        # Grid Customization
        if x_grid_spacing is not None:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_grid_spacing))
        if y_grid_spacing is not None:
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(y_grid_spacing))
        
        ax.grid(True, axis='x', color=GREY91, linestyle='-', linewidth=grid_linewidth)
        ax.grid(True, axis='y', color=GREY91, linestyle='-', linewidth=grid_linewidth)
        
        ax.xaxis.set_label_position('top')
        
        if y_lim:
            ax.set_ylim(y_lim)
        
        if x_lim:
            ax.set_xlim(x_lim)
        
        ax.set_xlabel('Time (Ma)',
                    labelpad=8.0,
                    fontname="Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )
        
        if y_variable == 'depth':
            y_label = 'Depth (m)'
        elif y_variable == 'temp':
            y_label = 'Temperature (ºC)'
        
        ax.set_ylabel(y_label,
                    fontname="Arial",
                    fontsize=12,
                    weight=500,
                    color=GREY40
                    )
        
        ax.tick_params(axis="x", length=2, color=GREY91)
        ax.tick_params(axis="y", length=2, color=GREY91)
        
        plt.xticks(fontsize=10, fontname='Arial', color=GREY40)
        plt.yticks(fontsize=10, fontname='Arial', color=GREY40)
        
        # Title
        is_master = sample_name == self.master_sample['sample_name']
        sample_type = "Master Sample" if is_master else "Sample"
        num_matched = len([p for p in matched_paths if p in list(good_paths.keys()) + list(acc_paths.keys())])
        
        title = f"{sample_type} {sample_name} - {y_variable.capitalize()} Space ({plot_type})\n"
        if is_master and y_variable == 'depth':
            title += f"{num_matched} paths match constraints"
        else:
            title += f"Paths matching master sample depth constraints"
        
        plt.title(title,
                fontsize=15,
                fontname="Arial",
                color=GREY10,
                pad=15)
        
        # Show and Save figure 
        if saveFig:
            pathlib.Path(saveFolder).mkdir(parents=True, exist_ok=True)
            if savefigFileName:
                filepath = f'{saveFolder}/{savefigFileName}.pdf'
            else:
                filepath = f'{saveFolder}/{sample_name}_{y_variable}_{plot_type}_matched_paths.pdf'
            plt.savefig(filepath, dpi='figure', bbox_inches='tight', pad_inches=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Print information about matched paths
        print(f"There are {num_matched} paths that meet these conditions.")
