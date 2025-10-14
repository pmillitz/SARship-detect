import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_vessel_length_histogram(df, save_path=None, suptitle=None):
    """
    Create a histogram showing vessel length distribution by class.
    """
    df_clean = df.dropna(subset=['vessel_length_m']).copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    
    vessel_data = df_clean[df_clean['is_vessel'] == True]
    fishing_data = df_clean[df_clean['is_fishing'] == True]
    
    # Create histogram with overlapping distributions, truncate bins at 400m
    bins = np.linspace(df_clean['vessel_length_m'].min(),
                       min(400, df_clean['vessel_length_m'].max()), 30)
    
    if len(vessel_data) > 0:
        ax.hist(vessel_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='red', label=f'is_vessel (n={len(vessel_data)})',
                edgecolor='darkred', linewidth=0.5)
    
    if len(fishing_data) > 0:
        ax.hist(fishing_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='lime', label=f'is_fishing (n={len(fishing_data)})',
                edgecolor='darkgreen', linewidth=0.5)
    
    ax.set_xlabel('Vessel length (m)', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    # No title - user will add figure caption in document
    ax.set_xlim(right=400)  # Truncate to eliminate extreme outliers
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add super title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

def plot_vessel_length_boxplot(df, save_path=None, suptitle=None):
    """
    Create a boxplot showing vessel length distribution by class.
    """
    df_clean = df.dropna(subset=['vessel_length_m']).copy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    vessel_data = df_clean[df_clean['is_vessel'] == True]
    fishing_data = df_clean[df_clean['is_fishing'] == True]

    # Prepare data for boxplot
    data_to_plot = []
    labels = []

    if len(vessel_data) > 0:
        data_to_plot.append(vessel_data['vessel_length_m'])
        labels.append('is_vessel')

    if len(fishing_data) > 0:
        data_to_plot.append(fishing_data['vessel_length_m'])
        labels.append('is_fishing')

    # Create boxplot with narrower boxes
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5),
                    widths=0.5)

    # Color the boxes to match histogram colors
    colors = ['red', 'lime']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Set median line color to dark blue
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(2)

    ax.set_ylabel('Vessel length (m)', fontsize=16)
    # No title - user will add figure caption in document
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add super title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

def plot_vessel_length_hist_and_box(df, save_path=None, suptitle=None):
    """
    Create side-by-side histogram and boxplot of vessel length distributions.
    """
    df_clean = df.dropna(subset=['vessel_length_m']).copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.25, 4.5), gridspec_kw={'width_ratios': [2, 1]})

    vessel_data = df_clean[df_clean['is_vessel'] == True]
    fishing_data = df_clean[df_clean['is_fishing'] == True]

    # ===== HISTOGRAM (LEFT PLOT) =====
    bins = np.linspace(df_clean['vessel_length_m'].min(),
                       min(400, df_clean['vessel_length_m'].max()), 30)

    if len(vessel_data) > 0:
        ax1.hist(vessel_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='red', label=f'is_vessel (n={len(vessel_data)})',
                edgecolor='darkred', linewidth=0.5)

    if len(fishing_data) > 0:
        ax1.hist(fishing_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='lime', label=f'is_fishing (n={len(fishing_data)})',
                edgecolor='darkgreen', linewidth=0.5)

    ax1.set_xlabel('Vessel length (m)', fontsize=16)
    ax1.set_ylabel('Frequency', fontsize=16)
    # No title - user will add figure caption in document
    ax1.set_xlim(right=400)  # Truncate to eliminate extreme outliers
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== BOXPLOT (RIGHT PLOT) =====
    data_to_plot = []
    labels = []

    if len(vessel_data) > 0:
        data_to_plot.append(vessel_data['vessel_length_m'])
        labels.append('is_vessel')

    if len(fishing_data) > 0:
        data_to_plot.append(fishing_data['vessel_length_m'])
        labels.append('is_fishing')

    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5),
                    widths=0.5)

    # Color the boxes to match histogram colors
    colors = ['red', 'lime']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Set median line color to dark blue
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(1)

    ax2.set_ylabel('Vessel length (m)', fontsize=16)
    # No title - user will add figure caption in document
    ax2.tick_params(axis='both', labelsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add super title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

# Example usage:
# plot_vessel_length_histogram(your_dataframe)
# plot_vessel_length_boxplot(your_dataframe)
# plot_vessel_length_hist_and_box(your_dataframe)
#
# With super titles:
# plot_vessel_length_histogram(your_dataframe, suptitle='Vessel Length Analysis - Training Data')
# plot_vessel_length_boxplot(your_dataframe, suptitle='Statistical Summary of Vessel Lengths')
# plot_vessel_length_hist_and_box(your_dataframe, suptitle='Complete Vessel Length Distribution Analysis')
#
# To save as PDF:
# plot_vessel_length_histogram(your_dataframe, save_path='histogram.pdf')
# plot_vessel_length_boxplot(your_dataframe, save_path='boxplot.pdf')
# plot_vessel_length_hist_and_box(your_dataframe, save_path='combined.pdf')
