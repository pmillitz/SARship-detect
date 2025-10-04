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
    
    # Create histogram with overlapping distributions
    bins = np.linspace(df_clean['vessel_length_m'].min(), 
                       df_clean['vessel_length_m'].max(), 30)
    
    if len(vessel_data) > 0:
        ax.hist(vessel_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='red', label=f'is_vessel (n={len(vessel_data)})',
                edgecolor='darkred', linewidth=0.5)
    
    if len(fishing_data) > 0:
        ax.hist(fishing_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='lime', label=f'is_fishing (n={len(fishing_data)})',
                edgecolor='darkgreen', linewidth=0.5)
    
    ax.set_xlabel('Vessel length (metres)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of vessel length by class')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add super title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # Add summary statistics as text centered below the legend
    if len(vessel_data) > 0 and len(fishing_data) > 0:
        vessel_median = vessel_data['vessel_length_m'].median()
        fishing_median = fishing_data['vessel_length_m'].median()
        vessel_iqr = vessel_data['vessel_length_m'].quantile(0.75) - vessel_data['vessel_length_m'].quantile(0.25)
        fishing_iqr = fishing_data['vessel_length_m'].quantile(0.75) - fishing_data['vessel_length_m'].quantile(0.25)
        vessel_min = vessel_data['vessel_length_m'].min()
        vessel_max = vessel_data['vessel_length_m'].max()
        fishing_min = fishing_data['vessel_length_m'].min()
        fishing_max = fishing_data['vessel_length_m'].max()

        stats_text = f'is_vessel:\n'
        stats_text += f'      median = {vessel_median:.1f}m\n'
        stats_text += f'      IQR    = {vessel_iqr:.1f}m\n'
        stats_text += f'      range  = {vessel_min:.1f}-{vessel_max:.1f}m\n'
        stats_text += f'is_fishing:\n'
        stats_text += f'      median = {fishing_median:.1f}m\n'
        stats_text += f'      IQR    = {fishing_iqr:.1f}m\n'
        stats_text += f'      range  = {fishing_min:.1f}-{fishing_max:.1f}m'

        ax.text(0.85, 0.75, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='center', fontsize=6,
                bbox=dict(boxstyle='round', edgecolor='black', facecolor='white', alpha=0.8, linewidth=1),
                multialignment='left')

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

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    vessel_data = df_clean[df_clean['is_vessel'] == True]
    fishing_data = df_clean[df_clean['is_fishing'] == True]

    # Prepare data for boxplot
    data_to_plot = []
    labels = []

    if len(vessel_data) > 0:
        data_to_plot.append(vessel_data['vessel_length_m'])
        labels.append(f'is_vessel (n={len(vessel_data)})')

    if len(fishing_data) > 0:
        data_to_plot.append(fishing_data['vessel_length_m'])
        labels.append(f'is_fishing (n={len(fishing_data)})')

    # Create boxplot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Color the boxes to match histogram colors
    colors = ['red', 'lime']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Set median line color to dark blue
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(2)

    ax.set_ylabel('Vessel length (metres)')
    ax.set_title('Boxplot of vessel length by class')
    ax.grid(True, alpha=0.3, axis='y')

    # Add super title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    # Add summary statistics as text at top center
    if len(vessel_data) > 0 and len(fishing_data) > 0:
        vessel_median = vessel_data['vessel_length_m'].median()
        fishing_median = fishing_data['vessel_length_m'].median()
        vessel_iqr = vessel_data['vessel_length_m'].quantile(0.75) - vessel_data['vessel_length_m'].quantile(0.25)
        fishing_iqr = fishing_data['vessel_length_m'].quantile(0.75) - fishing_data['vessel_length_m'].quantile(0.25)
        vessel_min = vessel_data['vessel_length_m'].min()
        vessel_max = vessel_data['vessel_length_m'].max()
        fishing_min = fishing_data['vessel_length_m'].min()
        fishing_max = fishing_data['vessel_length_m'].max()

        stats_text = f'is_vessel:\n'
        stats_text += f'      median = {vessel_median:.1f}m\n'
        stats_text += f'      IQR    = {vessel_iqr:.1f}m\n'
        stats_text += f'      range  = {vessel_min:.1f}-{vessel_max:.1f}m\n'
        stats_text += f'is_fishing:\n'
        stats_text += f'      median = {fishing_median:.1f}m\n'
        stats_text += f'      IQR    = {fishing_iqr:.1f}m\n'
        stats_text += f'      range  = {fishing_min:.1f}-{fishing_max:.1f}m'

        ax.text(0.46, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='center', fontsize=6,
                bbox=dict(boxstyle='round', edgecolor='black', facecolor='white', alpha=0.8, linewidth=1),
                wrap=True)

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
                       df_clean['vessel_length_m'].max(), 30)

    if len(vessel_data) > 0:
        ax1.hist(vessel_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='red', label=f'is_vessel (n={len(vessel_data)})',
                edgecolor='darkred', linewidth=0.5)

    if len(fishing_data) > 0:
        ax1.hist(fishing_data['vessel_length_m'], bins=bins, alpha=0.6,
                color='lime', label=f'is_fishing (n={len(fishing_data)})',
                edgecolor='darkgreen', linewidth=0.5)

    ax1.set_xlabel('Vessel length (metres)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of vessel length by class')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== BOXPLOT (RIGHT PLOT) =====
    data_to_plot = []
    labels = []

    if len(vessel_data) > 0:
        data_to_plot.append(vessel_data['vessel_length_m'])
        labels.append(f'is_vessel (n={len(vessel_data)})')

    if len(fishing_data) > 0:
        data_to_plot.append(fishing_data['vessel_length_m'])
        labels.append(f'is_fishing (n={len(fishing_data)})')

    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))

    # Color the boxes to match histogram colors
    colors = ['red', 'lime']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Set median line color to dark blue
    for median in bp['medians']:
        median.set_color('darkblue')
        median.set_linewidth(1)

    ax2.set_ylabel('Vessel Length (metres)')
    ax2.set_title('Boxplot of vessel length by class')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add summary statistics as text in the histogram (right of center)
    if len(vessel_data) > 0 and len(fishing_data) > 0:
        vessel_median = vessel_data['vessel_length_m'].median()
        fishing_median = fishing_data['vessel_length_m'].median()
        vessel_iqr = vessel_data['vessel_length_m'].quantile(0.75) - vessel_data['vessel_length_m'].quantile(0.25)
        fishing_iqr = fishing_data['vessel_length_m'].quantile(0.75) - fishing_data['vessel_length_m'].quantile(0.25)
        vessel_min = vessel_data['vessel_length_m'].min()
        vessel_max = vessel_data['vessel_length_m'].max()
        fishing_min = fishing_data['vessel_length_m'].min()
        fishing_max = fishing_data['vessel_length_m'].max()

        stats_text = f'is_vessel:\n'
        stats_text += f'    {"median":<6} = {vessel_median:.1f}m\n'
        stats_text += f'    {"IQR":<6} = {vessel_iqr:.1f}m\n'
        stats_text += f'    {"range":<6} = {vessel_min:.1f}-{vessel_max:.1f}m\n'
        stats_text += f'is_fishing:\n'
        stats_text += f'    {"median":<6} = {fishing_median:.1f}m\n'
        stats_text += f'    {"IQR":<6} = {fishing_iqr:.1f}m\n'
        stats_text += f'    {"range":<6} = {fishing_min:.1f}-{fishing_max:.1f}m'

        ax1.text(0.95, 0.75, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=11.5,
                multialignment='left', fontfamily='monospace',
                bbox=dict(boxstyle='round', edgecolor='black', facecolor='white', alpha=0.8, linewidth=1.7))

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
