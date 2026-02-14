#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pan-gene family analysis toolkit

Author: Xiaodong Li
Email: lxd1997xy@163.com
Version: 1.0.0
Date: 2026-01-07
License: BSD-3-Clause
"""

import os
import subprocess
import argparse
import shutil
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Phylo.PAML import yn00
from pathlib import Path
from scipy.optimize import curve_fit
from collections import defaultdict
import logging
import random
import tempfile

from tqdm import tqdm

from scipy.stats import norm
import tomli as toml
import random


def calculate_homo_kaks(args):
    """
    Optimized Kaks calculation worker
    args: (seq1_record, seq2_record)
    """
    s1, s2 = args
    g1, g2 = s1.id, s2.id
    
    # Validation
    if not s1 or not s2:
        return None
    
    # Using system temp dir to avoid I/O contention
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Translate sequences
            p1 = str(s1.seq.translate(table=1, cds=False)).rstrip('*')
            p2 = str(s2.seq.translate(table=1, cds=False)).rstrip('*')
            
            pair_dir = Path(temp_dir)
            
            # 1. Write Protein Fasta
            prot_fa = pair_dir / "pep.fa"
            cds_fa = pair_dir / "cds.fa"
            
            with open(prot_fa, "w") as f:
                f.write(f">seq1\n{p1}\n>seq2\n{p2}\n")
            with open(cds_fa, "w") as f:
                f.write(f">seq1\n{s1.seq}\n>seq2\n{s2.seq}\n")
            
            # 2. Alignment using Muscle v5 (avoid shell=True)
            aln_fa = pair_dir / "pep.aln"
            subprocess.run(
                ["muscle", "-align", str(prot_fa), "-output", str(aln_fa)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            # 3. Pal2Nal (Protein Alignment -> Codon Alignment)
            paml_in = pair_dir / "codon.phy"
            with open(paml_in, "w") as f_out:
                subprocess.run(
                    ["pal2nal.pl", str(aln_fa), str(cds_fa), "-output", "paml", "-nogap"],
                    check=True, stdout=f_out, stderr=subprocess.DEVNULL
                )
            
            # 4. YN00 Calculation (PAML)
            yn = yn00.Yn00()
            yn.alignment = str(paml_in)
            yn.out_file = str(pair_dir / "yn00.out")
            yn.working_dir = str(pair_dir)
            yn.set_options(verbose=0, icode=0, weighting=0, commonf3x4=0)
            
            res = yn.run(verbose=False)
            
            # Parse Result - flexible lookup
            pair_data = None
            if 'seq1' in res and 'seq2' in res['seq1']:
                pair_data = res['seq1']['seq2']
            elif 'seq2' in res and 'seq1' in res['seq2']:
                pair_data = res['seq2']['seq1']
            else:
                # Fallback deep search
                for k1 in res:
                    for k2 in res[k1]:
                        pair_data = res[k1][k2]
                        break
                    if pair_data:
                        break
            
            if pair_data:
                # Prefer NG86 method (same as KsPeakFitterv2)
                method_data = pair_data.get('NG86', pair_data.get('YN00'))
                
                if method_data:
                    dn = method_data.get('dN', 0)
                    ds = method_data.get('dS', 0)
                    omega = method_data.get('omega', 0)
                    
                    if ds is not None:
                        return {
                            'Gene1': g1,
                            'Gene2': g2,
                            'dN': float(dn) if dn is not None else 0,
                            'dS': float(ds),
                            'omega': float(omega) if omega is not None else 0
                        }
        except Exception:
            return None
    
    return None

class GeneFamilyVisualizer:
    def __init__(self, output_dir, config=None):
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "plots"
        self.assoc_dir = self.output_dir / "assoc"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.assoc_dir.mkdir(parents=True, exist_ok=True)
        self.config = config if config else {}
        self._set_plot_style()

    def _set_plot_style(self):
        """Sets matplotlib params for publication-quality figures."""
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['axes.linewidth'] = 1.0
        mpl.rcParams['xtick.major.width'] = 1.0
        mpl.rcParams['ytick.major.width'] = 1.0
        mpl.rcParams['font.size'] = 10 
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.titlesize'] = 14
        mpl.rcParams['savefig.dpi'] = 300

    def plot_pav_heatmap(self, df, genomes_cols, ogg_family_counts=None):
        """Generates a PAV heatmap with stacked bar charts."""
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import ListedColormap
        
        # Config
        cfg = self.config.get('plot', {}).get('pav_heatmap', {})
        figsize = cfg.get('figsize', [20, 8])
        pav_colors = cfg.get('pav_colors', ['#F0F0F0', '#B2182B']) # Optimized: Absent(Light), Present(Deep Red)
        class_colors = cfg.get('class_colors', {'Core': '#FE7270', 'Softcore': '#C68EC5', 'Shell': '#68D2D1', 'Specific': '#EADC94'})
        xlabel = cfg.get('xlabel', "Orthogroups")
        
        # Sort by Class then by Total count
        class_order = {'Core': 0, 'Softcore': 1, 'Shell': 2, 'Specific': 3}
        # Fill missing with 4 to ensure numeric comparison works (avoiding NaN != NaN causing many dividers)
        df['Class_Order'] = df['Class'].map(class_order).fillna(4)
        df_sorted = df.sort_values(['Class_Order', 'Total'], ascending=[True, False])
        
        # Calculate Class Boundaries for divider lines
        boundary_indices = []
        if not df_sorted.empty:
            current_cls = df_sorted.iloc[0]['Class_Order']
            for i in range(len(df_sorted)):
                if df_sorted.iloc[i]['Class_Order'] != current_cls:
                    boundary_indices.append(i)
                    current_cls = df_sorted.iloc[i]['Class_Order']
        
        # Prepare PAV matrix
        pav_matrix = (df_sorted[genomes_cols] > 0).astype(int).T.values
        
        # Prepare Right Bar Data (Gene counts per class per genome)
        class_counts = df.groupby('Class')[genomes_cols].sum().T
        desired_order = ['Core', 'Softcore', 'Shell', 'Specific']
        # Ensure all columns exist
        for c in desired_order:
            if c not in class_counts.columns: class_counts[c] = 0
        class_counts = class_counts[desired_order]
        
        # Prepare Top Bar Data
        if ogg_family_counts is not None:
            ogg_family_counts = ogg_family_counts.reindex(df_sorted['Orthogroup']).fillna(0)
        
        # Plotting
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, width_ratios=[15, 3], height_ratios=[3, 15], wspace=0.05, hspace=0.05)
        
        ax_top = fig.add_subplot(gs[0, 0])
        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])

        # Despine
        for ax in [ax_top, ax_right]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if ax == ax_top: 
                ax.spines['bottom'].set_visible(False)
        
        # 1. Heatmap
        cmap = ListedColormap(pav_colors) # Absent, Present
        ax_heatmap.imshow(pav_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Add white vertical dividers between classes
        for idx in boundary_indices:
             ax_heatmap.axvline(x=idx-0.5, color='white', linewidth=1.5)

        ax_heatmap.set_yticks(np.arange(len(genomes_cols)))
        ax_heatmap.set_yticklabels(genomes_cols)
        ax_heatmap.set_xlabel(xlabel)
        ax_heatmap.set_xticks([]) # Hide X Tick labels
        
        # 2. Top Bar
        if ogg_family_counts is not None:
            families = ogg_family_counts.columns
            bottom = np.zeros(len(df_sorted))
            # Colors for families
            if len(families) <= 10:
                fam_colors = plt.cm.tab10(np.arange(len(families)))
            else:
                fam_colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
                
            for i, fam in enumerate(families):
                values = ogg_family_counts[fam].values
                ax_top.bar(np.arange(len(df_sorted)), values, bottom=bottom, color=fam_colors[i], label=fam, width=1.0)
                bottom += values
            
            ax_top.set_xlim(-0.5, len(df_sorted)-0.5)
            ax_top.set_xticks([])
            ax_top.set_ylabel("Gene Count")
            ax_top.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), fontsize='small', frameon=False, ncol=len(families))
            
        # 3. Right Bar
        y_pos = np.arange(len(genomes_cols))
        left = np.zeros(len(genomes_cols))
        
        for cls in desired_order:
            values = class_counts[cls].values
            c_color = class_colors.get(cls, '#333333')
            ax_right.barh(y_pos, values, left=left, color=c_color, label=cls, height=0.9)
            left += values
            
        ax_right.set_ylim(-0.5, len(genomes_cols)-0.5)
        ax_right.invert_yaxis()
        ax_right.set_yticks([])
        ax_right.set_xlabel("Gene Count")
        
        # Formatting Right Bar
        ax_right.spines['bottom'].set_visible(True) # Show bottom axis
        # Move legend to top right outside to prevent overlap
        ax_right.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), fontsize='small', title="Class", frameon=False, ncol=2)

        plt.savefig(self.plot_dir / "PAV_heatmap_composite.pdf", bbox_inches='tight')
        plt.close()

    def plot_fit_pan(self, results):
        """Generates Pan/Core curve plots."""
        import matplotlib.ticker as ticker
        # Config
        cfg = self.config.get('plot', {}).get('fit_pan', {})
        figsize = cfg.get('figsize', [10, 6])
        colors = cfg.get('colors', {'pan': '#E64B35', 'core': '#4DBBD5'}) # Pan, Core
        markers = cfg.get('markers', {'pan': 'o', 'core': '^'})
        
        # User adjustable parameters for fitting
        pan_p0 = cfg.get('pan_p0', None) # Initial guess [a, b] for Power Law
        core_p0 = cfg.get('core_p0', None) # Initial guess [a, b, c] for Exp Decay
        
        pan_data = results['pan']
        core_data = results['core']
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Pan
        x_pan, y_pan = zip(*pan_data)
        ax.scatter(x_pan, y_pan, alpha=0.5, color=colors['pan'], marker=markers['pan'], s=30)
        
        # Fit Pan (Power Law: y = A * x^B)
        # Pan genome size should monotonically increase, so we constrain B > 0
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        try:
            # Prepare initial guess and bounds
            # Default guess: a = first value, b = 0.5 (square root growth)
            p0 = pan_p0 if pan_p0 else [y_pan[0], 0.5]
            # Bounds: a > 0, b > 0 (Strictly increasing)
            bounds = ([0, 0], [np.inf, np.inf])
            
            popt_pan, _ = curve_fit(power_law, x_pan, y_pan, p0=p0, bounds=bounds, maxfev=5000)
            
            x_fit = np.linspace(min(x_pan), max(x_pan), 100)
            y_fit = power_law(x_fit, *popt_pan)
            ax.plot(x_fit, y_fit, color=colors['pan'], label='Pan', linewidth=2)
            logging.info(f"Pan-genome fit (Power Law): y = {popt_pan[0]:.2f} * x^{popt_pan[1]:.2f}")
        except Exception as e:
            logging.warning(f"Pan-genome fitting failed: {e}")
            ax.plot(x_pan, y_pan, color=colors['pan'], label='Pan', linewidth=2)

        # Core
        x_core, y_core = zip(*core_data)
        ax.scatter(x_core, y_core, alpha=0.5, color=colors['core'], marker=markers['core'], s=30)
        
        # Fit Core (Exponential Decay: y = A * exp(B * x) + C)
        # Core genome size should decay, so B < 0
        def exp_decay(x, a, b, c):
            return a * np.exp(b * x) + c

        try:
            # Default guess: a = range, b = -0.1, c = min
            p0 = core_p0 if core_p0 else [max(y_core)-min(y_core), -0.1, min(y_core)]
            # Bounds: a>0, b<0, c>0
            bounds = ([0, -np.inf, 0], [np.inf, 0, np.inf])
            
            popt_core, _ = curve_fit(exp_decay, x_core, y_core, p0=p0, bounds=bounds, maxfev=5000)
            
            x_fit = np.linspace(min(x_core), max(x_core), 100)
            y_fit = exp_decay(x_fit, *popt_core)
            ax.plot(x_fit, y_fit, color=colors['core'], label='Core', linewidth=2)
            logging.info(f"Core-genome fit (Exp Decay): y = {popt_core[0]:.2f} * exp({popt_core[1]:.2f}x) + {popt_core[2]:.2f}")
        except Exception as e:
            logging.warning(f"Core-genome fitting failed: {e}")
            ax.plot(x_core, y_core, color=colors['core'], label='Core', linewidth=2)

        ax.set_xlabel("Number of Genomes")
        ax.set_ylabel("Number of Gene Families")
        
        # SCI Optimization
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / "pan_genome_fit.pdf")
        plt.close()

    def plot_duplication_stats(self, df):
        """Plots duplication types (WGD, TD, etc) by Class."""
        if df.empty: return

        # Pivot: Count of (Class, Type)
        # We want to see: For Core genes, how many are WGD, TD...
        counts = df.groupby(['Class', 'Type']).size().unstack(fill_value=0)
        
        # Reorder Index (Class)
        order_cls = ['Core', 'Softcore', 'Shell', 'Specific']
        existing_cls = [c for c in order_cls if c in counts.index]
        counts = counts.reindex(existing_cls)
        
        # Reorder Columns (Type)
        order_type = ['WGD', 'Tandem', 'Proximal', 'Transposed', 'Dispersed']
        # Include types present in data even if not in standard list
        existing_type = [t for t in order_type if t in counts.columns] + [t for t in counts.columns if t not in order_type]
        counts = counts[existing_type] 
        
        # Config
        cfg = self.config.get('plot', {}).get('duplication', {})
        figsize = cfg.get('figsize', [6, 3])
        type_colors = cfg.get('colors', {
            'WGD': '#E64B35', 'Tandem': '#4DBBD5', 'Proximal': '#00A087', 
            'Transposed': '#3C5488', 'Dispersed': '#F39B7F'
        })
        colors = [type_colors.get(t, '#808080') for t in counts.columns]
        
        # 1. Absolute Counts Plot
        fig, ax = plt.subplots(figsize=figsize)
        counts.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8, edgecolor='none')
        
        ax.set_ylabel("Number of Genes")
        ax.legend(title='Duplication', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.plot_dir / "duplication_modes_by_class.pdf", bbox_inches='tight')
        plt.close()
        
        # 2. Percentage Plot
        fig, ax = plt.subplots(figsize=figsize)
        counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100
        counts_pct.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8, edgecolor='none')
        
        ax.set_ylabel("Percentage (%)")
        ax.legend(title='Duplication', frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / "duplication_modes_by_class_pct.pdf", bbox_inches='tight')
        plt.close()

    def _perform_statistical_tests(self, data_dict, output_name):
        """
        Runs stats (T-test or ANOVA/Tukey) and returns annotation info.
        """
        if len(data_dict) < 2: return {}
        
        from scipy import stats
        
        results = []
        annotations = {}
        groups = list(data_dict.keys())
        
        # 2 Groups
        if len(groups) == 2:
            g1, g2 = groups
            try:
                # Calculate basic statistics
                n1, n2 = len(data_dict[g1]), len(data_dict[g2])
                mean1, mean2 = np.mean(data_dict[g1]), np.mean(data_dict[g2])
                std1, std2 = np.std(data_dict[g1], ddof=1), np.std(data_dict[g2], ddof=1)
                
                s, p = stats.ttest_ind(data_dict[g1], data_dict[g2], equal_var=False)
                sig = 'ns'
                if p < 0.001: sig = '***'
                elif p < 0.01: sig = '**'
                elif p < 0.05: sig = '*'
                
                results.append({
                    'Test': 'T-test', 
                    'Group1': g1,
                    'N1': n1,
                    'Mean1': mean1,
                    'Std1': std1,
                    'Group2': g2,
                    'N2': n2,
                    'Mean2': mean2,
                    'Std2': std2,
                    'P-value': p, 
                })
                annotations = {'type': 'stars', 'pair': (g1, g2), 'sig': sig}
            except Exception as e:
                logging.warning(f"Stats error: {e}")

        # >2 Groups
        else:
            try:
                vals = [data_dict[g] for g in groups]
                f, p = stats.f_oneway(*vals)
                
                # Calculate basic statistics for all groups
                group_stats = []
                for g in groups:
                    group_stats.append({
                        'Group': g,
                        'N': len(data_dict[g]),
                        'Mean': np.mean(data_dict[g]),
                        'Std': np.std(data_dict[g], ddof=1)
                    })
                
                if p < 0.05:
                    # Try Tukey
                    try:
                        from statsmodels.stats.multicomp import pairwise_tukeyhsd
                        # Flatten
                        endog = []
                        exog = []
                        for g, v in data_dict.items():
                            endog.extend(v)
                            exog.extend([g]*len(v))
                            
                        tukey = pairwise_tukeyhsd(endog, exog)
                        res_data = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                        
                        # Simple letter assignment (Greedy based on sorted means)
                        sorted_grps = sorted(groups, key=lambda g: np.mean(data_dict[g]), reverse=True)
                        current_char = 97 # 'a'
                        letter_map = {g: '' for g in groups}
                        letter_map[sorted_grps[0]] = chr(current_char)
                        
                        for i in range(1, len(sorted_grps)):
                            g_prev = sorted_grps[i-1]
                            g_curr = sorted_grps[i]
                            # Check pairwise
                            row = res_data[((res_data['group1']==g_prev) & (res_data['group2']==g_curr)) | ((res_data['group1']==g_curr) & (res_data['group2']==g_prev))]
                            if not row.empty and row.iloc[0]['reject']:
                                current_char += 1
                                letter_map[g_curr] = chr(current_char)
                            else:
                                letter_map[g_curr] = letter_map[g_prev]
                                
                        annotations = {'type': 'letters', 'map': letter_map}
                        
                        # Add Tukey post-hoc results with group statistics and ANOVA info
                        for _, r in res_data.iterrows():
                            g1_name, g2_name = r['group1'], r['group2']
                            g1_stats = next((gs for gs in group_stats if gs['Group'] == g1_name), None)
                            g2_stats = next((gs for gs in group_stats if gs['Group'] == g2_name), None)
                            
                            tukey_result = {
                                'Test': 'Tukey',
                                'Group1': g1_name,
                                'N1': g1_stats['N'] if g1_stats else np.nan,
                                'Mean1': g1_stats['Mean'] if g1_stats else np.nan,
                                'Std1': g1_stats['Std'] if g1_stats else np.nan,
                                'Group2': g2_name,
                                'N2': g2_stats['N'] if g2_stats else np.nan,
                                'Mean2': g2_stats['Mean'] if g2_stats else np.nan,
                                'Std2': g2_stats['Std'] if g2_stats else np.nan,
                                'Mean-diff': r['meandiff'],
                                'P-value': r['p-adj'], 
                                'ANOVA-F': f,
                                'ANOVA-P': p
                            }
                            results.append(tukey_result)
                            
                    except ImportError:
                        # If Tukey is not available, just report ANOVA
                        results.append({
                            'Comparison': 'Overall', 
                            'Test': 'ANOVA',
                            'Warning': 'statsmodels not found, skipping post-hoc',
                            'ANOVA-F': f,
                            'ANOVA-P': p,
                        })
                        annotations = {'type': 'anova_only', 'p': p}
                else:
                    # ANOVA not significant, no post-hoc needed
                    results.append({
                        'Comparison': 'Overall', 
                        'Test': 'ANOVA',
                        'ANOVA-F': f,
                        'ANOVA-P': p,
                    })
                    annotations = {}

            except Exception as e:
                logging.warning(f"ANOVA error: {e}")
        
        if results:
            pd.DataFrame(results).to_csv(self.plot_dir / f"{output_name}_stats.csv", index=False)
            
        return annotations

    def plot_expression_stats(self, data, genome_name):
        """
        Plots Expression Mean and Tau distribution (if available), grouping by Pan-Genome Class.
        """
        cfg = self.config.get('plot', {}).get('expression', {})
        figsize = cfg.get('figsize', [4, 3])
        plot_type = cfg.get('type', 'box') # box or violin
        base_colors = cfg.get('colors', ['#FE7270', '#C68EC5', '#68D2D1', '#EADC94', '#AAAAAA'])
        class_palette = {
            'Core': base_colors[0], 
            'Softcore': base_colors[1] if len(base_colors)>1 else base_colors[0],
            'Shell': base_colors[2] if len(base_colors)>2 else base_colors[0],
            'Specific': base_colors[3] if len(base_colors)>3 else base_colors[0],
            'Unknown': base_colors[4] if len(base_colors)>4 else '#CCCCCC'
        }
        
        df = pd.DataFrame(data)
        if df.empty: return

        order = ['Core', 'Softcore', 'Shell', 'Specific']
        present_classes = [c for c in order if c in df['Class'].unique()]
        if not present_classes: return

        # Define metrics to plot
        metrics = []
        if 'Exp_Mean' in df.columns:
            metrics.append({'col': 'Exp_Mean', 'ylabel': 'Log2(Mean Expression + 1)', 'suffix': '_exp_stats.pdf'})
        if 'Tau' in df.columns:
            metrics.append({'col': 'Tau', 'ylabel': 'Tissue Specificity Index (Tau)', 'suffix': '_tau_stats.pdf'})

        for m in metrics:
            col = m['col']
            ylabel = m['ylabel']
            suffix = m['suffix']
            
            # Prepare data
            data_dict = {c: df[df['Class'] == c][col].dropna().values for c in present_classes}
            data_dict = {k: v for k, v in data_dict.items() if len(v) > 0}
            present_classes_plot = [c for c in present_classes if c in data_dict]
            
            if not present_classes_plot: continue
            
            plot_data_list = [data_dict[c] for c in present_classes_plot]
            
            # Stats annotation
            stats_anno = self._perform_statistical_tests(data_dict, f"{genome_name}_{col}")

            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'violin':
                parts = ax.violinplot(plot_data_list, showmeans=False, showmedians=True)
                for i, pc in enumerate(parts['bodies']):
                    c_name = present_classes_plot[i]
                    pc.set_facecolor(class_palette.get(c_name, '#333333'))
                    pc.set_edgecolor('none')
                    pc.set_alpha(0.8)
                for partname in ('cbars','cmins','cmaxes','cmedians'):
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1)
            else:
                # Matplotlib 3.9+
                box = ax.boxplot(plot_data_list, patch_artist=True, tick_labels=present_classes_plot, medianprops=dict(color="black"))

                for i, patch in enumerate(box['boxes']):
                    c_name = present_classes_plot[i]
                    patch.set_facecolor(class_palette.get(c_name, '#333333'))
                    patch.set_edgecolor('none')
                    patch.set_alpha(0.9)
            
            # Parsing annotations
            if stats_anno:
                # Helper to find visible max (whisker top) to avoid outliers pushing labels too high
                def get_visible_max(d):
                    if len(d) == 0: return 0
                    q1, q3 = np.percentile(d, [25, 75])
                    iqr = q3 - q1
                    upper_fence = q3 + 1.5 * iqr
                    # Values displayed by default boxplot whiskers (without outliers)
                    within_fence = d[d <= upper_fence]
                    return np.max(within_fence) if len(within_fence) > 0 else np.max(d)

                # Used for finding Y range context
                y_concat = np.concatenate(plot_data_list)
                y_max_real = np.max(y_concat) if len(y_concat) > 0 else 0
                y_min_real = np.min(y_concat) if len(y_concat) > 0 else 0
                y_range_real = y_max_real - y_min_real
                if y_range_real == 0: y_range_real = 1.0
                
                # Let's recalculate visual max of the whole plot
                visual_maxs = [get_visible_max(d) for d in plot_data_list]
                visual_plot_max = max(visual_maxs) if visual_maxs else 0
                
                # We use visual_plot_max to determine annotation height, 
                # but we reference y_range_real for relative spacing size to keep it proportional 
                # (or use visual range if outliers are extreme).
                # Using visual range is safer for spacing.
                visual_mins = [np.min(d) for d in plot_data_list if len(d)>0]
                visual_plot_min = min(visual_mins) if visual_mins else 0
                visual_range = visual_plot_max - visual_plot_min
                if visual_range == 0: visual_range = 1.0
                
                top_margin_y = visual_plot_max
                
                if stats_anno.get('type') == 'stars':
                    g1, g2 = stats_anno['pair']
                    sig = stats_anno['sig']
                    if sig != 'ns':
                        idx1 = present_classes_plot.index(g1)
                        idx2 = present_classes_plot.index(g2)
                        
                        v1 = get_visible_max(data_dict[g1])
                        v2 = get_visible_max(data_dict[g2])
                        y_m = max(v1, v2)
                        
                        h = visual_range * 0.05
                        y_line = y_m + h
                        
                        # Draw bracket
                        ax.plot([idx1+1, idx1+1, idx2+1, idx2+1], [y_line, y_line+h, y_line+h, y_line], lw=1, c='black')
                        ax.text((idx1+idx2+2)/2, y_line+h, sig, ha='center', va='bottom')
                        top_margin_y = max(top_margin_y, y_line + h * 2.5)
                
                elif stats_anno.get('type') == 'letters':
                    pmap = stats_anno['map']
                    for i, cls in enumerate(present_classes_plot):
                        if cls in pmap:
                            y_val = get_visible_max(data_dict[cls])
                            ax.text(i+1, y_val + visual_range*0.05, pmap[cls], ha='center', va='bottom', fontweight='bold')
                            top_margin_y = max(top_margin_y, y_val + visual_range * 0.20)
                            
                elif stats_anno.get('type') == 'anova_only':
                    p_val = stats_anno.get('p', 1.0)
                    ax.set_title(f"ANOVA p={p_val:.2e}", fontsize='small')
                
                # Set ylim to focus on visible area + annotations
                # We ignore real extreme outliers in scaling if they are hidden
                ax.set_ylim(top=top_margin_y + visual_range * 0.05)

            ax.set_ylabel(ylabel)
            ax.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / f"{genome_name}{suffix}")
            plt.close()

    def plot_expression_heatmap(self, heatmap_df, gene_meta, genome_name):
        """
        Plots Tissue Expression Heatmap with Tau (Top) and Class Trends (Right).
        heatmap_df: Index=Gene, Columns=Tissues (Values: Log2 Expression)
        gene_meta: Index=Gene, Columns=[Class, Tau]
        """
        from matplotlib import gridspec
        
        # Align
        common_genes = heatmap_df.index.intersection(gene_meta.index)
        if len(common_genes) == 0: return
        
        heatmap_df = heatmap_df.loc[common_genes]
        gene_meta = gene_meta.loc[common_genes]
        
        # Sort Genes: Class (Core->Specific) then Tau (Desc)
        class_order = {'Core': 0, 'Softcore': 1, 'Shell': 2, 'Specific': 3}
        gene_meta['Class_Int'] = gene_meta['Class'].map(class_order).fillna(4)
        gene_meta.sort_values(['Class_Int', 'Tau'], ascending=[True, False], inplace=True)
        
        sorted_genes = gene_meta.index
        heatmap_df = heatmap_df.loc[sorted_genes]
        
        # Prepare Data Layout
        # Heatmap: Rows=Tissues, Cols=Genes
        data_matrix = heatmap_df.T # (n_tissues, n_genes)
        tissues = data_matrix.index
        
        # Top Data: Tau values
        tau_values = gene_meta.loc[sorted_genes, 'Tau'].values
        
        # Right Data: Class Trends (Average Expression per Class per Tissue)
        # We need a matrix: Rows=Tissues, Cols=Classes
        # Join heatmap_df (Genes x Tissues) with Class
        tmp = heatmap_df.join(gene_meta['Class'])
        class_trends = tmp.groupby('Class').mean().T # (n_tissues, n_classes)
        # Order classes
        existing_classes = [c for c in ['Core', 'Softcore', 'Shell', 'Specific'] if c in class_trends.columns]
        class_trends = class_trends[existing_classes]

        # Config
        cfg = self.config.get('plot', {}).get('expression_heatmap', {})
        # Reduce height by half (approx), keeping width
        figsize = cfg.get('figsize', (16, 6)) 
        # Make colormap more beautiful (e.g., YlGnBu or Spectra_r for expression)
        cmap = cfg.get('cmap', 'YlGnBu')
        colors_map = self.config.get('plot', {}).get('class_colors', {'Core': '#FE7270', 'Softcore': '#C68EC5', 'Shell': '#68D2D1', 'Specific': '#EADC94'})

        # GridSpec Structure
        # Top (Tau) | Empty (Legend)
        # Heatmap   | Right (Trend)
        # Height Ratios: 2 (Top), 10 (Heatmap)
        # Width Ratios: 10 (Heatmap), 2 (Right)
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, width_ratios=[10, 2], height_ratios=[2, 10], wspace=0.1, hspace=0.1)
        
        ax_top = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        # Add pure legend axis
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_legend.axis('off')

        # 1. Top Bar (Tau)
        ax_top.bar(np.arange(len(sorted_genes)), tau_values, width=1.0, color='#555555', alpha=0.8)
        ax_top.set_xlim(-0.5, len(sorted_genes)-0.5)
        ax_top.set_xticks([])
        ax_top.set_ylabel("Tau")
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        
        # 2. Main Heatmap
        im = ax_main.imshow(data_matrix.values, aspect='auto', cmap=cmap, interpolation='nearest')
        
        # Axis labels
        ax_main.set_yticks(np.arange(len(tissues)))
        ax_main.set_yticklabels(tissues, fontsize='small')
        ax_main.set_xlabel(f"Genes (Sorted by Class) n={len(sorted_genes)}")
        ax_main.set_xticks([])
        
        # Class Dividers (Vertical lines)
        current_cls_int = gene_meta['Class_Int'].iloc[0]
        for i in range(len(gene_meta)):
            if gene_meta['Class_Int'].iloc[i] != current_cls_int:
                ax_main.axvline(x=i-0.5, color='white', linewidth=1, linestyle='--')
                ax_top.axvline(x=i-0.5, color='black', linewidth=1, linestyle='--')
                current_cls_int = gene_meta['Class_Int'].iloc[i]
        
        # 3. Right Plot (Trends)
        y_pos = np.arange(len(tissues))
        handles = []
        labels = []
        for cls in existing_classes:
            vals = class_trends[cls].values
            color = colors_map.get(cls, 'black')
            if np.all(np.isnan(vals)) or np.sum(vals) == 0:
                continue
            line, = ax_right.plot(vals, y_pos, marker='o', markersize=4, label=cls, color=color, linewidth=2)
            handles.append(line)
            labels.append(cls)
            
        ax_right.set_ylim(-0.5, len(tissues)-0.5)
        ax_right.invert_yaxis() # Match Heatmap top-down
        ax_right.set_yticks([])
        ax_right.set_xlabel("Mean Exp")
        ax_right.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Only show bottom spine for trends
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.spines['bottom'].set_visible(True)

        # Place Legend in empty top-right space
        ax_legend.legend(handles, labels, loc='center', fontsize='small', frameon=False, title="Class")

        # Colorbar for Heatmap (Inset axes below heatmap)
        # bounds = [x, y, width, height] in transAxes
        ax_cbar = ax_main.inset_axes([0.25, -0.12, 0.5, 0.03])
        cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
        cbar.set_label('Log2(TPM+1)', fontsize='small')

        plt.savefig(self.plot_dir / f"{genome_name}_exp_heatmap_composite.pdf", bbox_inches='tight')
        plt.close()

    def plot_volcano(self, df, trait):
        """Plots volcano plot for univariate association analysis, distinguishing CNV and PAV."""
        cfg = self.config.get('plot', {}).get('volcano', {})
        figsize = cfg.get('figsize', [4, 4])
        markersize = cfg.get('markersize', 40)
        alpha = cfg.get('alpha', 0.6)
        p_threshold = cfg.get('p_threshold', 0.05)
        fdr_threshold = cfg.get('fdr_threshold', 0.1)
        
        # Default colors for CNV and PAV
        default_colors = {'CNV': '#E64B35', 'PAV': '#4DBBD5'}
        colors = cfg.get('colors', default_colors)
        
        # Calculate -log10(p)
        df['-log10(P)'] = -np.log10(df['P_value'].replace(0, 1e-300))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot by Matrix_Type if available
        if 'Matrix_Type' in df.columns:
            for matrix_type in ['CNV', 'PAV']:
                mask = df['Matrix_Type'] == matrix_type
                if mask.sum() > 0:
                    color = colors.get(matrix_type, '#999999')
                    ax.scatter(df.loc[mask, 'Effect_Size'], df.loc[mask, '-log10(P)'],
                              c=color, s=markersize, alpha=alpha, edgecolors='none',
                              label=matrix_type)
            ax.legend(fontsize=10, frameon=False, loc='best')
        else:
            # Fallback to single color
            ax.scatter(df['Effect_Size'], df['-log10(P)'],
                      c='#333333', s=markersize, alpha=alpha, edgecolors='none')
        
        # Add reference lines
        ax.axhline(-np.log10(p_threshold), color='gray', linestyle='--', linewidth=1, alpha=0.6,
                  label=f'$P={p_threshold}$')
        if 'FDR' in df.columns:
            ax.axhline(-np.log10(fdr_threshold), color='orange', linestyle='--', linewidth=1, alpha=0.6,
                      label=f'$\\mathrm{{FDR}}={fdr_threshold}$')
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
        
        ax.set_xlabel('Effect Size (Correlation)', fontsize=12)
        ax.set_ylabel('$-\\log_{10}(P)$', fontsize=12)
        ax.set_title(f'{trait}', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.assoc_dir / f"volcano_{trait}.pdf", bbox_inches='tight')
        plt.close()

    def plot_pav_boxplot(self, pav_data, pheno_data, pav_results, trait):
        """Plots boxplots for PAV associations (Absent vs Present) with significance annotation.
        
        Args:
            pav_data: PAV matrix (genomes x OGGs)
            pheno_data: Phenotype dataframe
            pav_results: DataFrame with PAV results ['Orthogroup', 'P_value', ...]
            trait: Trait name
        """
        from scipy.stats import ttest_ind
        
        cfg = self.config.get('plot', {}).get('pav_boxplot', {})
        figsize_per_plot = cfg.get('figsize_per_plot', [3, 3])
        plots_per_row = cfg.get('plots_per_row', 4)
        plot_type = cfg.get('type', 'box')
        show_points = cfg.get('show_points', True)
        point_size = cfg.get('point_size', 30)
        point_alpha = cfg.get('point_alpha', 0.5)
        pav_colors = cfg.get('pav_colors', ['#559AD2', '#FE7270'])
        
        if pav_results.empty:
            return
        
        n_plots = len(pav_results)
        n_rows = int(np.ceil(n_plots / plots_per_row))
        
        fig, axes = plt.subplots(n_rows, plots_per_row,
                                figsize=(figsize_per_plot[0] * plots_per_row,
                                        figsize_per_plot[1] * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(pav_results.iterrows()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            ogg = row['Orthogroup']
            
            if ogg not in pav_data.columns:
                ax.axis('off')
                continue
            
            pav_values = pav_data[ogg]
            pheno_values = pheno_data[trait]
            
            common_idx = pav_values.index.intersection(pheno_values.index)
            pav_aligned = pav_values.loc[common_idx]
            pheno_aligned = pheno_values.loc[common_idx]
            
            # Binary display for PAV
            data_absent = pheno_aligned[pav_aligned == 0].dropna()
            data_present = pheno_aligned[pav_aligned > 0].dropna()
            
            if len(data_absent) < 2 or len(data_present) < 2:
                ax.axis('off')
                continue
            
            # Check if data has sufficient variance (avoid numerical issues)
            if data_absent.std() < 1e-10 and data_present.std() < 1e-10:
                ax.axis('off')
                continue
            
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    _, pval = ttest_ind(data_absent, data_present, equal_var=False)
            except:
                pval = 1.0
            
            data_to_plot = [data_absent, data_present]
            labels = ['Absent', 'Present']
            
            if plot_type == 'violin':
                parts = ax.violinplot(data_to_plot, positions=[1, 2], showmeans=True)
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(pav_colors[i])
                    pc.set_alpha(0.6)
            else:
                bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                               widths=0.5, showfliers=False)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(pav_colors[i])
                    patch.set_alpha(0.7)
            
            if show_points:
                for i, data in enumerate(data_to_plot, 1):
                    x = np.random.normal(i, 0.04, len(data))
                    ax.scatter(x, data, s=point_size, alpha=point_alpha,
                              c=pav_colors[i-1], edgecolors='white', linewidth=0.5)
            
            # Add significance annotation with bracket
            y_max = max([d.max() for d in data_to_plot])
            y_min = min([d.min() for d in data_to_plot])
            y_range = y_max - y_min
            if y_range == 0:
                y_range = 1.0
            
            h = y_range * 0.05
            y_line = y_max + h
            ax.plot([1, 1, 2, 2], [y_line, y_line+h, y_line+h, y_line], 'k-', linewidth=1)
            
            # Significance stars
            if pval < 0.001:
                sig_text = '***'
            elif pval < 0.01:
                sig_text = '**'
            elif pval < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            ax.text(1.5, y_line+h, sig_text, ha='center', va='bottom', fontsize=10)
            ax.set_ylim(top=y_line + h * 2.5)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel(trait, fontsize=9)
            ax.set_title(f'{ogg}\n$P={pval:.2e}$', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.assoc_dir / f"pav_boxplot_{trait}.pdf", bbox_inches='tight')
        plt.close()
    
    def plot_cnv_scatter(self, cnv_data, pheno_data, cnv_results, trait):
        """Plots scatter plots for CNV associations (copy number vs phenotype).
        
        Args:
            cnv_data: CNV matrix (genomes x OGGs)
            pheno_data: Phenotype dataframe
            cnv_results: DataFrame with significant CNV associations
            trait: Trait name
        """
        from scipy.stats import spearmanr
        
        cfg = self.config.get('plot', {}).get('cnv_scatter', {})
        figsize_per_plot = cfg.get('figsize_per_plot', [3, 3])
        plots_per_row = cfg.get('plots_per_row', 4)
        color = cfg.get('color', '#E64B35')
        markersize = cfg.get('markersize', 60)
        alpha = cfg.get('alpha', 0.6)
        show_regline = cfg.get('show_regline', True)
        regline_color = cfg.get('regline_color', '#00468B')
        regline_width = cfg.get('regline_width', 2)
        
        if cnv_results.empty:
            return
        
        n_plots = len(cnv_results)
        n_rows = int(np.ceil(n_plots / plots_per_row))
        
        fig, axes = plt.subplots(n_rows, plots_per_row,
                                figsize=(figsize_per_plot[0] * plots_per_row,
                                        figsize_per_plot[1] * n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(cnv_results.iterrows()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            ogg = row['Orthogroup']
            
            if ogg not in cnv_data.columns:
                ax.axis('off')
                continue
            
            cnv_values = cnv_data[ogg]
            pheno_values = pheno_data[trait]
            
            common_idx = cnv_values.index.intersection(pheno_values.index)
            x = cnv_values.loc[common_idx].dropna()
            y = pheno_values.loc[x.index]
            
            if len(x) < 3:
                ax.axis('off')
                continue
            
            ax.scatter(x, y, s=markersize, alpha=alpha, c=color,
                      edgecolors='white', linewidth=0.5)
            
            # Add regression line
            if show_regline:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), color=regline_color,
                       linewidth=regline_width, linestyle='--', alpha=0.8)
            
            try:
                corr, pval = spearmanr(x, y)
            except:
                corr, pval = 0, 1.0
            
            ax.set_xlabel('Copy Number', fontsize=10)
            ax.set_ylabel(trait, fontsize=10)
            ax.set_title(f'{ogg}\n$r={corr:.3f}$, $P={pval:.2e}$', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.assoc_dir / f"cnv_scatter_{trait}.pdf", bbox_inches='tight')
        plt.close()

    def plot_ks_gmm(self, background_ks, family_ks, family_gene_counts, genome_name):
        """Plots WGD Ks distribution with GMM fitting and family member pie chart."""
        cfg = self.config.get('plot', {}).get('ks_gmm', {})
        figsize = cfg.get('figsize', [6, 4.5])
        n_components = cfg.get('n_components', 3)
        ks_min = cfg.get('ks_min', 0.0)
        ks_max = cfg.get('ks_max', 3.0)
        bins = cfg.get('bins', 100)
        colors = cfg.get('colors', ['#FE7270', '#C68EC5', '#68D2D1', '#EADC94', '#AAAAAA'])
        seed = cfg.get('seed', 42)
        
        if len(background_ks) == 0:
            logging.warning(f"No background Ks data for {genome_name}. Skipping GMM plot.")
            return
        
        # Filter background Ks
        bg_ks_filtered = background_ks[(background_ks >= ks_min) & (background_ks <= ks_max)]
        
        if len(bg_ks_filtered) < 10:
            logging.warning(f"Insufficient background Ks data ({len(bg_ks_filtered)}) for {genome_name}. Skipping GMM.")
            return
        
        # Fit GMM
        try:
            X = bg_ks_filtered.reshape(-1, 1)
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=seed)
            gmm.fit(X)
            
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            idx = np.argsort(means)
            means, stds, weights = means[idx], stds[idx], weights[idx]
            
            # Save GMM stats to plot_dir
            stats_df = pd.DataFrame({
                'Peak': range(1, n_components + 1),
                'Mean': means,
                'Std': stds,
                'Weight': weights
            })
            stats_path = self.plot_dir / f"{genome_name}_gmm_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            
        except Exception as e:
            logging.error(f"GMM fitting failed for {genome_name}: {e}")
            return
        
        # Classify family pairs into peaks
        pair_peak_counts = [0] * n_components
        if len(family_ks) > 0:
            fam_ks_filtered = family_ks[(family_ks >= ks_min) & (family_ks <= ks_max)]
            if len(fam_ks_filtered) > 0:
                X_fam = fam_ks_filtered.reshape(-1, 1)
                raw_probs = gmm.predict_proba(X_fam)
                sort_idx = np.argsort(gmm.means_.flatten())
                sorted_probs = raw_probs[:, sort_idx]
                assignments = np.argmax(sorted_probs, axis=1)
                pair_peak_counts = [np.sum(assignments == i) for i in range(n_components)]
        
        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        
        # Background histogram
        ax.hist(bg_ks_filtered, bins=bins, density=True, color='#e0e0e0', 
                edgecolor='none', alpha=0.5, label='Ks distribution')
        
        # GMM curves
        x_plot = np.linspace(ks_min, ks_max, 1000)
        for i in range(n_components):
            pdf = weights[i] * norm.pdf(x_plot, means[i], stds[i])
            c = colors[i % len(colors)]
            ax.plot(x_plot, pdf, '--', linewidth=1.2, color=c)
            ax.fill_between(x_plot, pdf, alpha=0.25, color=c, 
                           label=f'Peak {i+1} ($\\mu$={means[i]:.2f})')
        
        # Rug plot for family WGD
        rug_height = 0
        if len(family_ks) > 0:
            fam_ks_filtered = family_ks[(family_ks >= ks_min) & (family_ks <= ks_max)]
            if len(fam_ks_filtered) > 0:
                rug_height = 0.1
                ax.vlines(fam_ks_filtered, ymin=-rug_height, ymax=0, 
                         color='gray', lw=0.8, alpha=0.6, label='Family WGD')
        
        # Pie chart inset
        if sum(pair_peak_counts) > 0 and family_gene_counts[0] > 0:
            ax_pie = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
            pie_cols = [colors[i % len(colors)] for i in range(len(pair_peak_counts))]
            wedges, texts, autotexts = ax_pie.pie(
                pair_peak_counts, colors=pie_cols,
                autopct=lambda p: f'{p:.0f}%' if p > 5 else '',
                wedgeprops=dict(edgecolor='w', linewidth=1)
            )
            for txt in autotexts:
                txt.set_fontsize(8)
                txt.set_fontweight('bold')
            
            ax_pie.set_title("WGD Pairs", fontsize=8, fontweight='bold', pad=2)
            
            # Family gene counts below pie
            wgd_genes, total_genes = family_gene_counts
            ax_pie.text(0.5, -0.05, f"{wgd_genes}/{total_genes}",
                       ha='center', va='top', transform=ax_pie.transAxes,
                       fontsize=8, fontweight='bold')
        
        # Aesthetics
        ax.set_xlabel("$K_s$", fontsize=12, fontweight='bold')
        ax.set_ylabel("Density", fontsize=12, fontweight='bold')
        ax.set_xlim(ks_min, ks_max)
        ax.set_ylim(bottom=-rug_height)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        output_path = self.plot_dir / f"{genome_name}_ks_gmm.pdf"
        plt.savefig(output_path, transparent=True, dpi=300)
        plt.close()
        logging.info(f"Saved GMM plot to {output_path}")

    def plot_ortho_kaks_stats(self, df):
        """Generates Ks and Omega boxplots with outlier filtering."""
        # Config
        cfg = self.config.get('plot', {}).get('kaks', {})
        figsize = cfg.get('figsize', [4, 3]) # Adapted for individual plots
        colors_map = cfg.get('colors', {'Core': '#FE7270', 'Softcore': '#C68EC5', 'Shell': '#68D2D1', 'Specific': '#EADC94'})
        
        # Filtering thresholds (to exclude artifacts like 99.0 or errors)
        max_ks = cfg.get('max_ks', 5.0)      # Ks > 5 usually indicates saturation
        max_omega = cfg.get('max_omega', 3.0) # Omega > 3 is extremely rare/artifact
        
        order = ['Core', 'Softcore', 'Shell', 'Specific']
        existing_order = [c for c in order if c in df['Class'].unique()]
        
        # Check available columns. Data prep might have named it 'dS' or 'Ks'
        metrics = []
        if 'Ks' in df.columns:
            metrics.append({'col': 'Ks', 'limit': max_ks, 'ylabel': 'Ks', 'file': 'kaks_boxplot_Ks.pdf'})
        elif 'dS' in df.columns:
            metrics.append({'col': 'dS', 'limit': max_ks, 'ylabel': 'Ks (dS)', 'file': 'kaks_boxplot_Ks.pdf'})
            
        if 'omega' in df.columns:
            metrics.append({'col': 'omega', 'limit': max_omega, 'ylabel': 'Ka/Ks (ω)', 'file': 'kaks_boxplot_Omega.pdf'})
        
        for m in metrics:
            # Prepare data (filter abnormal)
            data_dict = {}
            for c in existing_order:
                subset = df[df['Class'] == c][m['col']]
                subset = pd.to_numeric(subset, errors='coerce').dropna()
                clean_subset = subset[(subset >= 0) & (subset <= m['limit'])].values
                if len(clean_subset) > 0:
                    data_dict[c] = clean_subset

            present_classes_plot = [c for c in existing_order if c in data_dict]
            
            if not present_classes_plot: continue
            
            data_to_plot = [data_dict[c] for c in present_classes_plot]
            labels = present_classes_plot
            
            # Stats
            stats_anno = self._perform_statistical_tests(data_dict, f"kaks_{m['col']}")

            fig, ax = plt.subplots(figsize=figsize)
            # Matplotlib 3.9+ renamed 'labels' to 'tick_labels'
            bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=labels, 
                           medianprops=dict(color="black"), showfliers=False)
            
            for patch, color_name in zip(bplot['boxes'], labels):
                patch.set_facecolor(colors_map.get(color_name, '#333333'))
                patch.set_edgecolor('none')
                patch.set_alpha(0.8)
            
            # Add Annotations
            if stats_anno:
                # Helper to find visible max (whisker top) to avoid outliers pushing labels too high
                def get_visible_max(d):
                    if len(d) == 0: return 0
                    q1, q3 = np.percentile(d, [25, 75])
                    iqr = q3 - q1
                    upper_fence = q3 + 1.5 * iqr
                    # Values displayed by default boxplot whiskers (without outliers)
                    within_fence = d[d <= upper_fence]
                    return np.max(within_fence) if len(within_fence) > 0 else np.max(d)

                y_concat = np.concatenate(data_to_plot)
                # Recalculate range based on visible part (since we hide fliers)
                visual_maxs = [get_visible_max(d) for d in data_to_plot]
                visual_plot_max = max(visual_maxs) if visual_maxs else 0
                visual_mins = [np.min(d) for d in data_to_plot if len(d)>0]
                visual_plot_min = min(visual_mins) if visual_mins else 0
                visual_range = visual_plot_max - visual_plot_min
                if visual_range == 0: visual_range = 1.0

                top_margin_y = visual_plot_max
                
                if stats_anno.get('type') == 'stars':
                    g1, g2 = stats_anno['pair']
                    sig = stats_anno['sig']
                    if sig != 'ns':
                        idx1 = labels.index(g1)
                        idx2 = labels.index(g2)
                        
                        v1 = get_visible_max(data_dict[g1])
                        v2 = get_visible_max(data_dict[g2])
                        y_m = max(v1, v2)
                        
                        h = visual_range * 0.05
                        y_line = y_m + h
                        # Draw bracket
                        ax.plot([idx1+1, idx1+1, idx2+1, idx2+1], [y_line, y_line+h, y_line+h, y_line], lw=1, c='black')
                        ax.text((idx1+idx2+2)/2, y_line+h, sig, ha='center', va='bottom')
                        top_margin_y = max(top_margin_y, y_line + h * 2.5)

                elif stats_anno.get('type') == 'letters':
                    pmap = stats_anno['map']
                    for i, cls in enumerate(labels):
                        if cls in pmap:
                            y_val = get_visible_max(data_dict[cls])
                            ax.text(i+1, y_val + visual_range*0.05, pmap[cls], ha='center', va='bottom', fontweight='bold')
                            top_margin_y = max(top_margin_y, y_val + visual_range * 0.20)
                            
                elif stats_anno.get('type') == 'anova_only':
                    p_val = stats_anno.get('p', 1.0)
                    ax.set_title(f"ANOVA p={p_val:.2e}", fontsize='small')

                # Auto expand ylim to prevent clipping
                ax.set_ylim(top=top_margin_y + visual_range * 0.05)

            ax.set_ylabel(m['ylabel'])
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            
            # Remove spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.plot_dir / m['file'])
            plt.close()
            logging.info(f"Generated {m['ylabel']} boxplot ({self.plot_dir / m['file']}) with threshold <= {m['limit']}")

    def plot_te_profile(self, profile_data, genome_name):
        """
        Plots TE density profile across Upstream-Body-Downstream.
        profile_data: dict { 'Class': np.array(90 floats) }
        The bins: 0-29 (Upstream 2kb), 30-59 (Body), 60-89 (Downstream 2kb)
        """
        cfg = self.config.get('plot', {}).get('te_profile', {})
        figsize = cfg.get('figsize', [4, 3])
        colors_map = cfg.get('colors', {'Core': '#FE7270', 'Softcore': '#C68EC5', 'Shell': '#68D2D1', 'Specific': '#EADC94'})
        linewidth = cfg.get('linewidth', 2)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Import smoothing function
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.interpolate import make_interp_spline
            has_scipy = True
        except ImportError:
            logging.warning("Scipy not found. TE profile smoothing disabled.")
            has_scipy = False

        x_vals = np.arange(90)
        
        for cls, values in profile_data.items():
            if np.all(values == 0): continue
            
            # Normalize to Density Distribution (Relative to Total)
            # This reflects the probability distribution along the gene body
            total_val = np.sum(values)
            if total_val > 0:
                values = values / total_val

            color = colors_map.get(cls, '#333333')
            
            # Smoothing strategy: Gaussian Filter -> Spline Interpolation
            # This prevents the "wiggly" line problem of pure spline on noisy data
            if has_scipy:
                try:
                    # 1. Gaussian Smoothing (reduces noise/jitter)
                    # sigma=2 usually gives a nice smooth trend for n=90 bins
                    values_smooth = gaussian_filter1d(values, sigma=2.0)
                    
                    # 2. Spline Interpolation (increases resolution for rendering)
                    x_new = np.linspace(x_vals.min(), x_vals.max(), 300)
                    spl = make_interp_spline(x_vals, values_smooth, k=3)
                    y_smooth = spl(x_new)
                    
                    # density cannot be negative
                    y_smooth[y_smooth < 0] = 0 
                    
                    ax.plot(x_new, y_smooth, color=color, label=cls, linewidth=linewidth)
                    ax.fill_between(x_new, y_smooth, color=color, alpha=0.2)
                except Exception as e:
                    logging.warning(f"Smoothing failed for {cls}: {e}. Plotting raw data.")
                    ax.plot(x_vals, values, color=color, label=cls, linewidth=linewidth)
                    ax.fill_between(x_vals, values, color=color, alpha=0.2)
            else:
                ax.plot(x_vals, values, color=color, label=cls, linewidth=linewidth)
                ax.fill_between(x_vals, values, color=color, alpha=0.2)
            
        # Add visual separators
        ax.axvline(x=29.5, color='gray', linestyle='--', alpha=0.5) # TSS boundary
        ax.axvline(x=59.5, color='gray', linestyle='--', alpha=0.5) # TTS boundary
        
        # Custom Ticks
        # 0 -> -2kb, 30 -> TSS, 60 -> TTS, 89 -> +2kb
        ax.set_xticks([0, 30, 60, 89])
        ax.set_xticklabels(["-2kb", "TSS", "TTS", "+2kb"])
        
        ax.set_ylabel("TE Density")
        ax.legend(frameon=False)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"{genome_name}_TE_density_profile.pdf")
        plt.close()

    def plot_family_counts(self, df_counts, tree_file):
        """
        Plots a combined figure: Phylogenetic Tree (Left) + Family Count Heatmap (Right).
        df_counts: DataFrame with index=Genome and columns=Family (counts)
        tree_file: Path to the newick tree file
        """
        from Bio import Phylo
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as ticker
        import matplotlib.colors as mcolors
        
        # Config
        plot_cfg = self.config.get('plot', {}).get('family_counts', {})
        show_tip_labels = plot_cfg.get('show_tip_labels', False)
        ladderize = plot_cfg.get('ladderize', True)
        cmap_name = plot_cfg.get('cmap', 'YlGnBu')
        show_cell_text = plot_cfg.get('show_cell_text', True)
        show_grid = plot_cfg.get('grid', True)
        figsize = plot_cfg.get('figsize', (18, 10))
        width_ratios = plot_cfg.get('width_ratios', [1, 1])
        colorbar_pad = plot_cfg.get('colorbar_pad', 0.3) # Pad for colorbar vs right-side labels
        inverted_scale = plot_cfg.get('inverted_scale', True) # Invert X-axis scale (tips=0)
        show_branch_support = plot_cfg.get('show_branch_support', False) # Show support values on node

        if not Path(tree_file).exists():
             logging.warning(f"Tree file {tree_file} not found. Skipping tree plot.")
             return

        # Load Tree
        try:
            tree = Phylo.read(tree_file, "newick")
        except Exception as e:
            logging.error(f"Failed to read tree file: {e}")
            return
            
        # Pruning: Keep only genomes present in stats
        genome_ids = set(df_counts.index)
        terms_to_prune = [t for t in tree.get_terminals() if t.name not in genome_ids]
        
        if len(terms_to_prune) == len(tree.get_terminals()):
             logging.warning("No matching genomes between tree and count data.")
             return

        for term in terms_to_prune:
            tree.prune(term)
        
        # Ladderize (sort branches by size)
        if ladderize:
            tree.ladderize()

        # Get leaf order (Phylo.draw usually plots 1..N bottom-up)
        # We need Top -> Bottom order for Heatmap
        leaves = [term.name for term in tree.get_terminals()]
        # Reverse to match Heatmap (Top=Row0 to Bottom=RowN)
        df_ordered = df_counts.reindex(leaves[::-1])
        
        # Plot Setup
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios, wspace=0.02)
        
        ax_tree = fig.add_subplot(gs[0])
        ax_heatmap = fig.add_subplot(gs[1])
        
        # --- 1. Plot Tree ---
        # Hide Spines but keep Bottom X-axis
        for sp in ax_tree.spines.values():
            sp.set_visible(False)
        ax_tree.spines['bottom'].set_visible(True)
        ax_tree.get_yaxis().set_visible(False)
        
        # Draw
        def get_label(x):
            # Only show labels for tip nodes (genomes) if enabled
            if show_tip_labels and x.is_terminal():
                return str(x.name)
            # Show internal node support if enabled
            if show_branch_support and not x.is_terminal():
                 if x.confidence is not None: return f"{x.confidence:.2f}"
                 if x.name and str(x.name).strip(): return str(x.name)
            return None

        Phylo.draw(tree, axes=ax_tree, do_show=False, show_confidence=show_branch_support, branch_labels=None, label_func=get_label)
        
        # X-axis (Evolutionary Distance or Time)
        terminals = tree.get_terminals()
        if terminals:
            max_depth = max(tree.distance(tree.root, t) for t in terminals)
        else:
            max_depth = 0
            
        ax_tree.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
        if inverted_scale:
            # Chronogram mode: Tips = 0 (Present), Root = Max Time (Past)
            # We keep the tree drawing standard (Left->Right), but invert the labels
            def reverse_time_formatter(x, pos):
                val = max_depth - x
                if abs(val) < 1e-4: return "0"
                return f"{val:.1f}"
                
            ax_tree.xaxis.set_major_formatter(ticker.FuncFormatter(reverse_time_formatter))
            ax_tree.set_xlabel("Divergence Time (MYA)")
            ax_tree.set_xlim(0, max_depth) 
        else:
            # Standard: Root = 0, Tips = Distance
            ax_tree.set_xlabel("Evolutionary Distance")
            ax_tree.set_xlim(0, max_depth)
        
        # Adjust Y limits (1-based to 0-based alignment)
        ax_tree.set_ylim(0.5, len(leaves) + 0.5)

        # --- 2. Heatmap ---
        data = df_ordered.values
        cax = ax_heatmap.imshow(data, cmap=cmap_name, aspect='auto')
        
        # Annotations (Text)
        if show_cell_text and df_ordered.size < 400:
             # Calculate contrast color
             norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
             cmap_obj = plt.get_cmap(cmap_name)
             
             for i in range(len(df_ordered)):
                 for j in range(len(df_ordered.columns)):
                     val = data[i, j]
                     rgba = cmap_obj(norm(val))
                     # Luminance
                     lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                     txt_color = 'white' if lum < 0.5 else 'black'
                     
                     ax_heatmap.text(j, i, str(int(val)),
                                     ha="center", va="center", color=txt_color, fontsize='small')
                                     
        # Axis Labels
        ax_heatmap.set_yticks(np.arange(len(df_ordered)))
        ax_heatmap.set_yticklabels(df_ordered.index)
        ax_heatmap.yaxis.tick_right() # Labels on right
        ax_heatmap.tick_params(axis='both', which='both', length=0) # Hide tick marks
        
        ax_heatmap.set_xticks(np.arange(len(df_ordered.columns)))
        ax_heatmap.set_xticklabels(df_ordered.columns, rotation=45, ha="left") # Ticks on bottom
        
        ax_heatmap.set_xlabel("Gene Families")
        
        # Grid (White)
        if show_grid:
            ax_heatmap.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
            ax_heatmap.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
            ax_heatmap.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
            ax_heatmap.tick_params(which="minor", bottom=False, left=False)

        # Remove Heatmap spines
        for sp in ax_heatmap.spines.values():
            sp.set_visible(False)
            
        fig.colorbar(cax, ax=ax_heatmap, label='Gene Count', shrink=0.5, pad=colorbar_pad)

        plt.savefig(self.plot_dir / "phylo_family_heatmap.pdf", bbox_inches='tight')
        plt.close()


class GeneFamilyAnalyzer:
    def __init__(self, input_dir, output_dir, hmm_configs, cds_dir=None, loc_dir=None, tree=None, 
                 threads=4, force=False, config=None,
                 phe_file=None, search_method='hmm'):
        """
        Initialize the Gene Family Analyzer.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.hmm_configs = hmm_configs
        self.cds_dir = Path(cds_dir) if cds_dir else None
        self.loc_dir = Path(loc_dir) if loc_dir else None
        self.tree = Path(tree) if tree else None
        
        self.threads = str(threads)
        self.force = force
        self.config = config if config else {}
        self.search_method = search_method

        # Default search parameters from config
        defaults = self.config.get('family', {}).get('search', {})
        
        # HMM defaults
        self.hmm_evalue = float(defaults.get('hmm_evalue', 1e-5))
        self.dom_evalue = float(defaults.get('dom_evalue', 10))
        self.dom_coverage = float(defaults.get('dom_coverage', 0.0))
        
        # BLAST defaults
        self.blast_evalue = float(defaults.get('blast_evalue', 1e-10))
        self.blast_identity = float(defaults.get('blast_identity', 50.0))
        self.blast_coverage = float(defaults.get('blast_coverage', 50.0))

        # Phenotype association configuration
        self.phe_file = Path(phe_file) if phe_file else None
        pheno_conf = self.config.get('analysis', {}).get('pheno_assoc', {})
        self.cnv_assoc = pheno_conf.get('cnv_assoc', True)
        self.pav_assoc = pheno_conf.get('pav_assoc', True)
        self.cnv_method = pheno_conf.get('cnv_method', 'spearman')
        self.pav_method = pheno_conf.get('pav_method', 'point_biserial')
        self.cnv_cap = pheno_conf.get('cnv_cap', None)
        
        # Initialize Visualizer
        self.visualizer = GeneFamilyVisualizer(output_dir, config)
        
        # Check Dependencies
        self.check_dependencies()
        
        # Ensure the main output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal Directory Structure
        self.family_root = self.output_dir / "family"
        self.genome_root = self.output_dir / "genomes"
        self.ortho_root = self.output_dir / "ortho"
        self.assoc_root = self.output_dir / "assoc"
        
        self.family_root.mkdir(exist_ok=True)
        self.genome_root.mkdir(exist_ok=True)
        self.assoc_root.mkdir(exist_ok=True)
        
        # Global accumulators for super-tree
        self.all_families_sequences = []
        self.all_families_stats = []
        # Store sequences by genome for OrthoFinder: {genome_name: [SeqRecords]}
        self.genome_merged_sequences = defaultdict(list)
        # Accumulator for final summary counts {(Genome, Family): count}
        self.summary_counts = defaultdict(int)
        
        # Mapping Gene -> Class (Core, Softcore, Shell, Specific)
        self.gene_class_map = {}

    def check_dependencies(self):
        """Check if required tools are in the system PATH."""
        required_tools = ["hmmsearch", "mafft", "orthofinder", "meme", "yn00", "pal2nal.pl", "diamond"]
        missing = [tool for tool in required_tools if not shutil.which(tool)]
        
        if missing:
            logging.error(f"The following required tools are missing from PATH: {', '.join(missing)}")
            logging.error("Please install them or add them to your environment path.")
            sys.exit(1)
        
        # Check optional phylogenetic tools based on config
        phylo_config = self.config.get('analysis', {}).get('phylogenetic', {})
        aligner = phylo_config.get('aligner', 'mafft')
        tree_builder = phylo_config.get('tree_builder', 'iqtree')
        
        optional_tools = []
        if aligner == 'muscle':
            optional_tools.append('muscle')
        if phylo_config.get('trim_alignment', True):
            optional_tools.append('trimal')
        if tree_builder == 'iqtree':
            optional_tools.append('iqtree')
        elif tree_builder == 'fasttree':
            optional_tools.append('fasttree')
        
        missing_optional = [tool for tool in optional_tools if not shutil.which(tool)]
        if missing_optional:
            logging.warning(f"Optional phylogenetic tools not found: {', '.join(missing_optional)}. Using defaults.")

    def run_command(self, cmd, shell=True, capture_output=False):
        """Helper function to execute shell commands cleanly."""
        try:
            if capture_output:
                subprocess.run(cmd, shell=shell, check=True)
            else:
                subprocess.run(cmd, shell=shell, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing command: {cmd}")
            if e.stderr:
                logging.error(f"Details: {e.stderr.decode()}")
            sys.exit(1)

    def get_proteome_files(self):
        """Identify proteome sequence files."""
        extensions = ('.fa', '.fasta', '.faa', '.pep')
        return [f for f in self.input_dir.iterdir() if f.suffix.lower() in extensions and f.is_file()]

    def parse_domtbl_with_domains(self, domtbl_path, param_dict=None):
        """
        Parses HMMER domtblout. Returns dict of gene IDs and domain details.
        Includes filtering by domain coverage and i-Evalue.
        """
        # Determine thresholds
        d_cov = self.dom_coverage
        d_evalue = self.dom_evalue
        
        if param_dict:
             hmm_conf = param_dict.get('hmm', {})
             d_cov = float(hmm_conf.get('dom_coverage', d_cov))
             d_evalue = float(hmm_conf.get('dom_evalue', d_evalue))

        hits_data = defaultdict(list)
        try:
            with open(domtbl_path, 'r') as f:
                for line in f:
                    if line.startswith("#"): continue
                    parts = line.split()
                    if len(parts) < 23: continue

                    gene_id = parts[0]
                    domain_name = parts[3]
                    hmm_len = int(parts[5])
                    i_evalue = float(parts[12])
                    start = int(parts[17])
                    end = int(parts[18])
                    
                    # Phase 1: Filtering
                    dom_len = end - start + 1
                    coverage = dom_len / hmm_len if hmm_len > 0 else 0
                    
                    if coverage < d_cov:
                        continue
                    if i_evalue > d_evalue:
                        continue

                    hits_data[gene_id].append({
                        'domain': domain_name,
                        'start': start,
                        'end': end,
                        'coverage': coverage,
                        'i_evalue': i_evalue
                    })
        except FileNotFoundError:
            pass
        return dict(hits_data)

    def parse_blast_output(self, file_path):
        """
        Parse BLAST fmt 6 output.
        Returns dict {gene_id: [{'domain', 'start', 'end', 'i_evalue'}]}
        """
        hits_data = defaultdict(list)
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split('\t')
                    if len(parts) < 12: continue
                    
                    query_id = parts[0]
                    subject_id = parts[1]
                    try:
                        s_start = int(parts[8])
                        s_end = int(parts[9])
                        evalue = float(parts[10])
                        
                        if s_start > s_end:
                            s_start, s_end = s_end, s_start

                        hits_data[subject_id].append({
                            'domain': query_id,
                            'start': s_start,
                            'end': s_end,
                            'i_evalue': evalue,
                            'coverage': 0.0
                        })
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
        return dict(hits_data)

    def extract_and_rename(self, proteome_path, hits_data, genome_name, fam_name):
        """
        Extracts sequences, renames them, calculates Physicochemical properties, and compiles stats.
        """
        seqs_objects = []
        mapping_data = []
        stats_data = []
        
        # Use index for large files (Phase 2: I/O Optimization)
        prot_dict = SeqIO.index(str(proteome_path), "fasta")
        sorted_original_ids = sorted(hits_data.keys())
        
        for i, original_id in enumerate(sorted_original_ids, 1):
            if original_id not in prot_dict: continue

            new_id = f"{genome_name}.{fam_name}.{i}"
            original_seq_record = prot_dict[original_id]
            seq_str = str(original_seq_record.seq).replace("*", "") # Remove stop chars for analysis
            seq_len = len(seq_str)
            
            # Phase 3: Physicochemical Properties
            try:
                pa = ProteinAnalysis(seq_str)
                mw = pa.molecular_weight()
                pi = pa.isoelectric_point()
            except Exception:
                mw = 0
                pi = 0
            
            new_seq_record = original_seq_record[:]
            new_seq_record.id = new_id
            new_seq_record.description = ""
            seqs_objects.append(new_seq_record)
            
            domain_list = sorted(hits_data[original_id], key=lambda x: x['start'])
            dom_names_str = ";".join([d['domain'] for d in domain_list])
            locs_str = ";".join([f"{max(1, d['start'])}-{min(seq_len, d['end'])}" for d in domain_list])

            mapping_data.append({"Original_ID": original_id, "New_ID": new_id})
            stats_data.append({
                "Family": fam_name,
                "Genome": genome_name,
                "Original_ID": original_id,
                "New_ID": new_id,
                "Length": seq_len,
                "Mw": round(mw, 2),
                "pI": round(pi, 2),
                "Domains": dom_names_str,
                "Locations": locs_str
            })
            
        return seqs_objects, pd.DataFrame(mapping_data), stats_data

    def fit_pan(self, df, genomes_cols, n_perm=10):
        """
        Calculates Pan and Core genome sizes for 1 to N genomes.
        Returns: {'pan': [(n, count)...], 'core': [(n, count)...]}
        """
        pav = (df[genomes_cols] > 0).astype(int)
        n_genomes = len(genomes_cols)
        
        pan_curve = []
        core_curve = []
        
        # For each number of genomes k
        for k in range(1, n_genomes + 1):
            temp_pan = []
            temp_core = []
            
            # Permutations
            for _ in range(n_perm):
                sampled_genomes = random.sample(genomes_cols, k)
                sub_pav = pav[sampled_genomes]
                
                # Pan: Genes present in ANY of the sampled genomes (sum > 0)
                n_pan = (sub_pav.sum(axis=1) > 0).sum()
                
                # Core: Genes present in ALL of the sampled genomes (sum == k)
                n_core = (sub_pav.sum(axis=1) == k).sum()
                
                temp_pan.append(n_pan)
                temp_core.append(n_core)
            
            # Record all points (visualizer will handle scatter, fitting handles mean)
            for val in temp_pan:
                pan_curve.append((k, val))
            for val in temp_core:
                core_curve.append((k, val))
                
        return {'pan': pan_curve, 'core': core_curve}

    def calculate_pairwise_kaks(self, g1, g2, cds_index):
        """
        Helper to run yn00 for a pair of genes.
        Requires creating temporary alignment files.
        """
        # 1. Get Sequences
        s1 = cds_index[g1]
        s2 = cds_index[g2]
        
        # 2. Protein Alignment (Translate first)
        try:
            p1 = s1.seq.translate(table=1, cds=False)
            p2 = s2.seq.translate(table=1, cds=False)
        except Exception:
            return None

        # Write temp protein fasta
        tmp_dir = self.ortho_root / "tmp_kaks"
        tmp_dir.mkdir(exist_ok=True)
        
        prot_fa = tmp_dir / "pair.prot.fa"
        with open(prot_fa, "w") as f:
            f.write(f">seq1\n{p1}\n>seq2\n{p2}\n")
            
        # Align proteins (using mafft)
        aln_fa = tmp_dir / "pair.aln.fa"
        subprocess.run(f"mafft --quiet {prot_fa} > {aln_fa}", shell=True)
        
        # 3. Map back to CDS
        if shutil.which("pal2nal.pl"):
            cds_fa = tmp_dir / "pair.cds.fa"
            with open(cds_fa, "w") as f:
                f.write(f">seq1\n{s1.seq}\n>seq2\n{s2.seq}\n")
            
            paml_in = tmp_dir / "yn00.nuc"
            cmd = f"pal2nal.pl {aln_fa} {cds_fa} -output paml -nogap > {paml_in}"
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                return None
        else:
            return None

        # 4. Run yn00
        ctl_file = tmp_dir / "yn00.ctl"
        out_file = tmp_dir / "yn00.out"
        with open(ctl_file, "w") as f:
            f.write(f"seqfile = {paml_in}\noutfile = {out_file}\nverbose = 0\nicode = 0\nweighting = 0\ncommonf3x4 = 0\n")
        
        try:
            subprocess.run(f"yn00 {ctl_file}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            res = yn00.read(str(out_file))
            for k1 in res:
                for k2 in res[k1]:
                    data = res[k1][k2]
                    if 'dN' in data and 'dS' in data:
                        return {'dN': data['dN'], 'dS': data['dS'], 'omega': data['omega']}
        except Exception:
            return None
            
        return None

    def build_phylogeny(self, fasta_path, output_dir, base_name):
        """Performs Multiple Sequence Alignment and Phylogenetic Tree Construction.
        
        Workflow:
        1. Alignment: MAFFT (default) or MUSCLE
        2. Trimming: TrimAl (optional, removes poorly aligned regions)
        3. Tree building: IQ-TREE (default, auto model selection) or FastTree
        """
        # Get phylogenetic config
        phylo_config = self.config.get('analysis', {}).get('phylogenetic', {})
        aligner = phylo_config.get('aligner', 'mafft')
        trim_alignment = phylo_config.get('trim_alignment', True)
        tree_builder = phylo_config.get('tree_builder', 'iqtree')
        trimal_method = phylo_config.get('trimal_method', 'automated1')
        iqtree_model = phylo_config.get('iqtree_model', 'MFP')  # Model Finder Plus
        iqtree_bootstrap = phylo_config.get('iqtree_bootstrap', 1000)
        
        msa_path = output_dir / f"{base_name}.aligned.fa"
        tree_path = output_dir / f"{base_name}.tree"
        
        # Step 1: Multiple Sequence Alignment
        logging.info(f"  Building phylogeny for {base_name}...")
        if aligner == 'muscle' and shutil.which('muscle'):
            logging.info(f"    [1/3] Aligning with MUSCLE...")
            logging.info(f"      muscle -align {fasta_path} -output {msa_path}")
            subprocess.run(
                ["muscle", "-align", str(fasta_path), "-output", str(msa_path)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            # Default to MAFFT
            logging.info(f"    [1/3] Aligning with MAFFT...")
            logging.info(f"      mafft --thread {self.threads} --auto --quiet {fasta_path} > {msa_path}")
            subprocess.run(
                f"mafft --thread {self.threads} --auto --quiet {fasta_path} > {msa_path}",
                shell=True, check=True
            )
        
        # Step 2: Trim Alignment (Optional)
        final_msa = msa_path
        if trim_alignment and shutil.which('trimal'):
            trimmed_path = output_dir / f"{base_name}.trimmed.fa"
            logging.info(f"    [2/3] Trimming alignment with TrimAl (-{trimal_method})...")
            logging.info(f"      trimal -in {msa_path} -out {trimmed_path} -{trimal_method}")
            subprocess.run(
                ["trimal", "-in", str(msa_path), "-out", str(trimmed_path), f"-{trimal_method}"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            final_msa = trimmed_path
        else:
            logging.info(f"    [2/3] Skipping alignment trimming.")
        
        # Step 3: Phylogenetic Tree Construction
        if tree_builder == 'iqtree' and shutil.which('iqtree'):
            logging.info(f"    [3/3] Building tree with IQ-TREE (model: {iqtree_model}, bootstrap: {iqtree_bootstrap})...")
            iqtree_prefix = output_dir / base_name
            cmd = [
                "iqtree",
                "-s", str(final_msa),
                "-m", iqtree_model,
                "-bb", str(iqtree_bootstrap),
                "-nt", str(self.threads),
                "-pre", str(iqtree_prefix),
                "-quiet"
            ]
            logging.info(f"      {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # IQ-TREE outputs .treefile, rename to .tree for consistency
            iqtree_output = output_dir / f"{base_name}.treefile"
            if iqtree_output.exists():
                iqtree_output.rename(tree_path)
        else:
            # Fallback to FastTree
            logging.info(f"    [3/3] Building tree with FastTree...")
            logging.info(f"      fasttree -wag -gamma -fastest {final_msa} > {tree_path}")
            subprocess.run(
                f"fasttree -wag -gamma -fastest {final_msa} > {tree_path}",
                shell=True, check=True
            )
        
        logging.info(f"    Phylogeny saved to {tree_path}")

    def run_meme(self, fasta_path, output_dir):
        """Runs MEME motif analysis."""
        output_dir.mkdir(exist_ok=True)
        cmd = f"meme {fasta_path} -oc {output_dir} -protein -nmotifs 10 -minw 6 -maxw 50"
        logging.info(f"Running MEME for {fasta_path.stem}...")
        try:
            self.run_command(cmd)
        except Exception as e:
            logging.error(f"MEME failed: {e}")

    def write_family_counts(self):
        """Writes the final count table to CSV in root output dir."""
        data = []
        for (genome, family), count in self.summary_counts.items():
            data.append({"Genome": genome, "Family": family, "Count": count})
        
        if len(data) > 0:
            df = pd.DataFrame(data)
            matrix_df = df.pivot(index='Genome', columns='Family', values='Count').fillna(0).astype(int)
            matrix_df.to_csv(self.output_dir / "summary_counts.csv")

            # If Phylogenetic Tree is provided, plot aligned heatmap
            if self.tree and self.tree.exists():
                logging.info("Plotting Family Count Heatmap with Phylogenetic Tree...")
                self.visualizer.plot_family_counts(matrix_df, self.tree)

    def build_ogg_tree(self, df):
        """Extracts representative sequences for each OGG and builds a tree."""
        logging.info("Building Orthogroup Tree...")

        # Check if tree already exists
        tree_file = self.ortho_root / "OGG_reps.tree"
        if not self.force and tree_file.exists():
            logging.info(f"  [Skip] OGG Tree found in {tree_file}")
            return

        og_seqs_dir = self.ortho_root / "Orthogroups" / "Orthogroup_Sequences"
        if not og_seqs_dir.exists():
            # Try finding it
            found = list(self.ortho_root.glob("Results_*/Orthogroups/Orthogroup_Sequences"))
            if found: 
                og_seqs_dir = found[0]
            else: 
                return

        reps = []
        for og_id in df['Orthogroup']:
            og_fa = og_seqs_dir / f"{og_id}.fa"
            if not og_fa.exists(): continue
            
            # Find longest seq
            longest_rec = None
            max_len = 0
            for rec in SeqIO.parse(og_fa, "fasta"):
                if len(rec.seq) > max_len:
                    max_len = len(rec.seq)
                    longest_rec = rec
            
            if longest_rec:
                longest_rec.id = og_id # Rename to OGG ID
                longest_rec.description = f"Representative from {longest_rec.description}"
                reps.append(longest_rec)
        
        if reps:
            rep_fa = self.ortho_root / "all_OGG_reps.fa"
            SeqIO.write(reps, rep_fa, "fasta")
            self.build_phylogeny(rep_fa, self.ortho_root, "OGG_reps")

    def run_orthofinder(self):
        """
        Orthology Detection (OrthoFinder).
        """
        # Check if OrthoFinder has already run
        ortho_results = list(self.ortho_root.glob("Results_*"))
        if not self.force and ortho_results:
            logging.info(f"  [Skip] OrthoFinder results found in {ortho_results[0]}")
        else:
            logging.info("Starting OrthoFinder analysis...")
            ortho_cmd = f"orthofinder -f {self.genome_root} -o {self.ortho_root} -t {self.threads}"
            logging.info(f"Executing: {ortho_cmd}")
            self.run_command(ortho_cmd, capture_output=True)

    def run_pan_genome(self):
        """
        Pan-Genome Classification and OGG Tree.
        """
        og_counts_file = self.ortho_root / "Orthogroups" / "Orthogroups.GeneCount.tsv"
        if not og_counts_file.exists():
            # Try finding it in subfolders if OrthoFinder structure varies
            # Optimized: Search in Results_* first
            found = list(self.ortho_root.glob("Results_*/Orthogroups/Orthogroups.GeneCount.tsv"))
            if found:
                og_counts_file = found[0]
            else:
                logging.warning("Orthogroups.GeneCount.tsv not found. Skipping Pan-Genome analysis.")
                return

        logging.info("Starting Pan-genome analysis...")
        df = pd.read_csv(og_counts_file, sep='\t')
        genomes_cols = [c for c in df.columns if c not in ['Orthogroup', 'Total']]
        num_genomes = len(genomes_cols)
        
        # Classification
        def classify(row):
            present_count = sum(row[genomes_cols] > 0)
            ratio = present_count / num_genomes
            
            if ratio == 1.0: return "Core"
            elif ratio >= 0.9: return "Softcore"
            elif present_count == 1: return "Specific"
            else: return "Shell"

        og_class_file = self.ortho_root / "pan_genome_classification.csv"
        df['Class'] = df.apply(classify, axis=1)

        # Save full classification results
        logging.info("Saving Pan-Genome Classification results...")
        df.to_csv(og_class_file, index=False)

        # Save stats
        stats = df['Class'].value_counts().reset_index()
        stats.columns = ['Type', 'Count']
        stats.to_csv(self.ortho_root / "pan_genome_stats.csv", index=False)

        # Prepare Family Counts for Visualization
        gene_family_map = {item['New_ID']: item['Family'] for item in self.all_families_stats}
        og_file = self.ortho_root / "Orthogroups" / "Orthogroups.tsv"
        if not og_file.exists():
            # Optimized: Search in Results_* first
            found = list(self.ortho_root.glob("Results_*/Orthogroups/Orthogroups.tsv"))
            if found: og_file = found[0]

        ogg_family_df = None
        if og_file.exists():
            ogg_class_dict = df.set_index('Orthogroup')['Class'].to_dict()
            og_tsv_df = pd.read_csv(og_file, sep='\t')
            
            # Vectorized processing to replace slow iterrows
            # 1. Melt to long format: [Orthogroup, Genome, GeneStr]
            genome_cols_in_tsv = [c for c in og_tsv_df.columns if c != 'Orthogroup']
            melted = og_tsv_df.melt(id_vars=['Orthogroup'], value_vars=genome_cols_in_tsv, value_name='Genes').dropna()
            
            if not melted.empty:
                # 2. Explode genes (split by ", ")
                # Ensure string type before splitting
                melted['Gene'] = melted['Genes'].astype(str).str.split(', ')
                exploded = melted.explode('Gene')[['Orthogroup', 'Gene']]
                
                # 3. Populate Gene -> Class Map (Bulk update)
                # Map Orthogroup to Class
                exploded['Class'] = exploded['Orthogroup'].map(ogg_class_dict).fillna("Unknown")
                
                # Update gene_class_map with all genes found
                # Using zip is much faster than iterating
                self.gene_class_map.update(dict(zip(exploded['Gene'], exploded['Class'])))
                
                # Update all_families_stats with OGG and Class info immediately
                gene_ogg_map = dict(zip(exploded['Gene'], exploded['Orthogroup']))
                if self.all_families_stats:
                    for item in self.all_families_stats:
                        nid = item['New_ID']
                        item['OGG'] = gene_ogg_map.get(nid, "-")
                        item['Class'] = self.gene_class_map.get(nid, "-")
                    
                    # Save updated stats
                    all_families_stats_path = self.output_dir / "all_family_stats.tsv"
                    pd.DataFrame(self.all_families_stats).to_csv(all_families_stats_path, sep='\t', index=False)
                    logging.info(f"Updated {all_families_stats_path} with OGG and Class info.")
                
                # 4. Count Specific Gene Families
                # Identify which genes belong to our families of interest
                exploded['Family'] = exploded['Gene'].map(gene_family_map)
                
                # Filter for genes that have a Family assignment and count
                fam_counts = exploded.dropna(subset=['Family']).groupby(['Orthogroup', 'Family']).size()
                
                if not fam_counts.empty:
                    # Unstack to get [Orthogroup x Family] matrix
                    ogg_family_df = fam_counts.unstack(fill_value=0)
                    # Reindex ensures we have all Orthogroups (even those with 0 counts) 
                    # matching the original file order
                    ogg_family_df = ogg_family_df.reindex(og_tsv_df['Orthogroup']).fillna(0).astype(int)
                else:
                    ogg_family_df = pd.DataFrame(index=og_tsv_df['Orthogroup'])
            else:
                ogg_family_df = pd.DataFrame(index=og_tsv_df['Orthogroup'])

            # Save Family Counts for Visualization
            logging.info("Saving Pan-Genome Family Counts...")
            ogg_family_df.to_csv(self.ortho_root / "pan_genome_family_counts.csv")

        # Visualization
        self.visualizer.plot_pav_heatmap(df, genomes_cols, ogg_family_df)
        
        # Calculate and plot saturation
        saturation_results = self.fit_pan(df, genomes_cols)
        self.visualizer.plot_fit_pan(saturation_results)

        # Build OGG Tree
        self.build_ogg_tree(df)

    def run_duplication_analysis(self):
        """
        Duplication Analysis using DupGen_finder.
        """
        logging.info("Starting Duplication Analysis...")
        # Check if tree already exists
        dup_summary = self.output_dir / "duplication_summary.csv"
        if not self.force and dup_summary.exists() and dup_summary.stat().st_size > 0:
            logging.info(f"  [Skip] Duplication summary found in {dup_summary}")
            return

        # 1. Check if loc_dir provided
        if not self.loc_dir or not self.loc_dir.exists():
            logging.info("  [Skip] No --loc directory provided for duplication analysis.")
            return
            
        # 2. Config Check for Outgroup
        dup_conf = self.config.get('analysis', {}).get('duplication', {})
        if not dup_conf:
            logging.info("  [Skip] No [analysis.duplication] configuration found.")
            return

        outgroup_gff = dup_conf.get('outgroup_gff')  # Already in DupGen format
        outgroup_pep = dup_conf.get('outgroup_pep')
        out_abbrev = dup_conf.get('outgroup_abbrev', 'Out')
        
        if not outgroup_gff or not outgroup_pep or not Path(outgroup_gff).exists() or not Path(outgroup_pep).exists():
            logging.warning("  [Skip] Duplication analysis: Outgroup GFF/PEP missing or invalid.")
            return

        # Check Tools
        diamond_cmd = shutil.which("diamond")
        dupgen_cmd = shutil.which("DupGen_finder.pl")
        if not dupgen_cmd: dupgen_cmd = shutil.which("DupGen_finder")
        
        if not diamond_cmd or not dupgen_cmd:
            logging.warning("  [Skip] Missing 'diamond' or 'DupGen_finder.pl'. Please install them.")
            return

        logging.info("Starting Duplication analysis...")
        dup_output_root = self.output_dir / "duplication_analysis"
        dup_output_root.mkdir(exist_ok=True)
        
        # Load Family Stats mapping
        if not hasattr(self, 'all_families_stats') or not self.all_families_stats:
            logging.warning("  [Skip] Family stats missing. Cannot map DupGen results.")
            return
        orig_to_new = {str(item['Original_ID']): item['New_ID'] for item in self.all_families_stats}

        # 3. Iterate BED files in loc_dir
        bed_files = list(self.loc_dir.glob("*.bed")) + list(self.loc_dir.glob("*.bed6"))
        if not bed_files:
            logging.warning(f"  [Skip] No BED files found in {self.loc_dir}")
            return
            
        final_results = []
        
        for bed_file in bed_files:
            # Extract genome name from filename (e.g., ZS11.bed -> ZS11)
            gname = bed_file.stem
            
            # Try to find corresponding PEP file
            pep_candidates = list(self.input_dir.glob(f"{gname}*.pep")) + \
                           list(self.input_dir.glob(f"{gname}*.fa")) + \
                           list(self.input_dir.glob(f"{gname}*.fasta"))
            
            if not pep_candidates:
                logging.warning(f"  [Skip] No PEP file found for {gname}")
                continue
                
            pep_file = pep_candidates[0]
            
            # Auto-generate abbreviation from genome name (first 3 characters)
            tgt_abbrev = gname[:3].capitalize() if len(gname) >= 3 else gname
            
            logging.info(f"  Running Duplication Analysis for {gname} (Abbrev: {tgt_abbrev})...")
            
            # Working Dirs
            work_dir = dup_output_root / gname
            data_dir = work_dir / "data"
            results_dir = work_dir / "results"
            
            if work_dir.exists() and not self.force:
                shutil.rmtree(work_dir) 
            
            work_dir.mkdir(exist_ok=True)
            data_dir.mkdir(exist_ok=True)
            results_dir.mkdir(exist_ok=True)
            
            # --- Step A: Prepare Files ---
            # 1. Convert Target Annotation (BED6 -> DupGen format)
            try:
                with open(bed_file, 'r') as fin, open(data_dir / f"{tgt_abbrev}.gff", 'w') as fout:
                    for line in fin:
                        if line.startswith('#') or not line.strip(): continue
                        parts = line.strip().split('\t')
                        # BED6: chrom, start, end, name, score, strand
                        if len(parts) >= 4:
                            chrom = parts[0]
                            # BED is 0-based, DupGen is 1-based
                            try:
                                start = int(parts[1]) + 1 
                                end = int(parts[2])
                                gene_id = parts[3]
                                fout.write(f"{tgt_abbrev}-{chrom}\t{gene_id}\t{start}\t{end}\n")
                            except ValueError:
                                continue
            except Exception as e:
                logging.error(f"Error converting {bed_file} to DupGen format: {e}")
                continue
            
            # 2. Prepare Outgroup Annotation (Direct Copy)
            try:
                shutil.copy(outgroup_gff, data_dir / f"{out_abbrev}.gff")
            except Exception as e:
                logging.error(f"    Failed to copy outgroup GFF: {e}")
                continue
            
            # 3. Merge Target and Outgroup Annotations
            try:
                merged_gff = data_dir / f"{tgt_abbrev}_{out_abbrev}.gff"
                with open(merged_gff, 'w') as fout:
                    with open(data_dir / f"{tgt_abbrev}.gff", 'r') as f1:
                        fout.write(f1.read())
                    with open(data_dir / f"{out_abbrev}.gff", 'r') as f2:
                        fout.write(f2.read())
            except Exception as e:
                logging.error(f"    Failed to merge GFF files: {e}")
                continue
            
            # 4. Copy Proteins
            try:
                shutil.copy(pep_file, data_dir / f"{tgt_abbrev}.pep")
                shutil.copy(outgroup_pep, data_dir / f"{out_abbrev}.pep")
            except Exception as e:
                logging.error(f"    Failed to copy protein files: {e}")
                continue

            # --- Step B: Diamond BLAST ---
            cwd = os.getcwd()
            os.chdir(data_dir) 
            
            try:
                # Make DBs
                if not Path(f"{tgt_abbrev}.dmnd").exists():
                    self.run_command(f"diamond makedb --in {tgt_abbrev}.pep --db {tgt_abbrev} --quiet")
                if not Path(f"{out_abbrev}.dmnd").exists():
                    self.run_command(f"diamond makedb --in {out_abbrev}.pep --db {out_abbrev} --quiet")
                
                # 1. Target vs Target (Self)
                if not Path(f"{tgt_abbrev}.blast").exists():
                    self.run_command(f"diamond blastp -q {tgt_abbrev}.pep -d {tgt_abbrev} -o {tgt_abbrev}.blast --outfmt 6 --evalue 1e-10 --max-target-seqs 5 --quiet")
                
                # 2. Target vs Outgroup
                pair_name = f"{tgt_abbrev}_{out_abbrev}.blast"
                if not Path(pair_name).exists():
                    self.run_command(f"diamond blastp -q {tgt_abbrev}.pep -d {out_abbrev} -o {pair_name} --outfmt 6 --evalue 1e-10 --max-target-seqs 5 --quiet")
            
            except Exception as e:
                logging.error(f"    BLAST failed: {e}")
                os.chdir(cwd)
                continue
            
            os.chdir(cwd)
            
            # --- Step C: Run DupGen_finder ---
            cmd = f"{dupgen_cmd} -i {data_dir.resolve()} -t {tgt_abbrev} -c {out_abbrev} -o {results_dir.resolve()}"
            logging.info(f"    Executing DupGen_finder...")
            try:
                self.run_command(cmd, capture_output=False)
            except Exception:
                logging.error(f"    DupGen_finder execution failed for {gname}")
                continue
                
            # --- Step D: Parse Results ---
            types = ['wgd', 'tandem', 'proximal', 'transposed', 'dispersed', 'singletons']
            
            logging.info(f"    Parsing DupGen results for {gname}...")
            for dtype in types:
                fpath = results_dir / f"{tgt_abbrev}.{dtype}.genes"
                if fpath.exists():
                    try:
                        with open(fpath, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if not parts: continue
                                gene = parts[0]
                                
                                # Map back to our IDs
                                if gene in orig_to_new:
                                    nid = orig_to_new[gene]
                                    cls = self.gene_class_map.get(nid, "Unknown")
                                    dname = 'WGD' if dtype == 'wgd' else dtype.capitalize()
                                    final_results.append({
                                        'Genome': gname,
                                        'Gene_Orig': gene,
                                        'Gene_New': nid,
                                        'Type': dname,
                                        'Class': cls
                                    })
                    except Exception as e:
                        logging.warning(f"    Error reading result {fpath}: {e}")
            
            # --- Step E: Ks Calculation for WGD Pairs (if CDS available) ---
            if self.cds_dir and dup_conf.get('calc_ks', True):
                wgd_pairs_file = results_dir / f"{tgt_abbrev}.wgd.pairs"
                kaks_output = work_dir / f"{gname}.wgd.kaks.csv"
                
                if wgd_pairs_file.exists():
                    # Check if calculation already done
                    if kaks_output.exists() and kaks_output.stat().st_size > 0 and not self.force:
                        logging.info(f"    [Skip] Ks calculation (Result exists: {kaks_output})")
                        try:
                            ks_df = pd.read_csv(kaks_output)
                        except Exception as e:
                            logging.warning(f"    Failed to load existing Ks results: {e}")
                            ks_df = None
                    else:
                        # Find CDS file for this genome
                        cds_candidates = list(self.cds_dir.glob(f"{gname}*.cds")) + \
                                       list(self.cds_dir.glob(f"{gname}*.fa")) + \
                                       list(self.cds_dir.glob(f"{gname}*.fasta"))
                        
                        if not cds_candidates:
                            logging.warning(f"    No CDS file found for {gname}. Skipping Ks calculation.")
                            ks_df = None
                        else:
                            cds_file = cds_candidates[0]
                            logging.info(f"    Calculating Ks for WGD pairs using {cds_file.name}...")
                            
                            # Parse WGD pairs (columns 0 and 2)
                            wgd_pairs = []
                            try:
                                with open(wgd_pairs_file, 'r') as f:
                                    for line in f:
                                        if line.startswith('#') or not line.strip():
                                            continue
                                        parts = line.strip().split('\t')
                                        if len(parts) >= 3:
                                            gene1 = parts[0]  # Duplicate 1
                                            gene2 = parts[2]  # Duplicate 2
                                            wgd_pairs.append((gene1, gene2))
                                logging.info(f"      Identified {len(wgd_pairs)} WGD pairs")
                            except Exception as e:
                                logging.error(f"    Failed to parse WGD pairs file: {e}")
                                wgd_pairs = []
                                ks_df = None
                            
                            if wgd_pairs:
                                # Load CDS sequences
                                try:
                                    cds_index = SeqIO.to_dict(SeqIO.parse(cds_file, "fasta"))
                                    logging.info(f"      Loaded {len(cds_index)} CDS sequences")
                                except Exception as e:
                                    logging.error(f"    Failed to load CDS file: {e}")
                                    cds_index = {}
                                    ks_df = None
                                
                                if cds_index:
                                    # Build task list
                                    tasks = []
                                    
                                    for g1, g2 in wgd_pairs:
                                        if g1 in cds_index and g2 in cds_index:
                                            tasks.append((cds_index[g1], cds_index[g2]))
                                    
                                    logging.info(f"      Computing Ks for {len(tasks)} valid pairs with {self.threads} threads...")
                                    
                                    # Parallel execution using multiprocessing.Pool (faster than ProcessPoolExecutor)
                                    results = []
                                    if tasks:
                                        from multiprocessing import Pool
                                        with Pool(processes=int(self.threads)) as pool:
                                            # Use imap_unordered with chunksize for better performance
                                            iterator = tqdm(pool.imap_unordered(calculate_homo_kaks, tasks, chunksize=20),
                                                          total=len(tasks), desc=f"    Ks calc ({gname})", unit="pair")
                                            
                                            for res in iterator:
                                                if res:
                                                    results.append(res)
                                    
                                    if results:
                                        ks_df = pd.DataFrame(results)
                                        ks_df.to_csv(kaks_output, index=False)
                                        logging.info(f"      Saved {len(ks_df)} Ks results to {kaks_output}")
                                    else:
                                        logging.warning(f"      No valid Ks results obtained")
                                        ks_df = None
                                else:
                                    ks_df = None
                            else:
                                ks_df = None
                    
                    # --- Step F: GMM Visualization ---
                    if ks_df is not None and not ks_df.empty:
                        logging.info(f"    Generating GMM visualization for {gname}...")
                        
                        # Extract Ks values (use dS column)
                        if 'dS' in ks_df.columns:
                            background_ks = ks_df['dS'].dropna().values
                        else:
                            logging.warning(f"    No 'dS' column in Ks results")
                            background_ks = np.array([])
                        
                        if len(background_ks) > 0:
                            # Filter for family members belonging to THIS genome only
                            genome_family_genes = set()
                            genome_total_count = 0
                            
                            if hasattr(self, 'all_families_stats') and self.all_families_stats:
                                for item in self.all_families_stats:
                                    if item.get('Genome') == gname:
                                        genome_family_genes.add(str(item['Original_ID']))
                                        genome_total_count += 1
                            
                            if genome_family_genes:
                                family_pairs = ks_df[
                                    (ks_df['Gene1'].isin(genome_family_genes)) & 
                                    (ks_df['Gene2'].isin(genome_family_genes))
                                ]
                                
                                if not family_pairs.empty and 'dS' in family_pairs.columns:
                                    family_ks = family_pairs['dS'].dropna().values
                                    
                                    # Count unique family genes in WGD
                                    family_wgd_genes = set(family_pairs['Gene1']).union(set(family_pairs['Gene2']))
                                    wgd_gene_count = len(family_wgd_genes)
                                    
                                    family_gene_counts = (wgd_gene_count, genome_total_count)
                                else:
                                    family_ks = np.array([])
                                    family_gene_counts = (0, genome_total_count)
                            else:
                                family_ks = np.array([])
                                family_gene_counts = (0, 0)
                            
                            # Generate plot (saved to plot_dir)
                            self.visualizer.plot_ks_gmm(
                                background_ks, family_ks, family_gene_counts, gname
                            )
                        else:
                            logging.warning(f"    No valid Ks values for GMM plotting")
                else:
                    logging.info(f"    WGD pairs file not found: {wgd_pairs_file}")
        
        # Save and Plot
        if not final_results:
             logging.info("  No duplication results collected.")
             return
             
        df_dups = pd.DataFrame(final_results)
        df_dups.to_csv(self.output_dir / "duplication_summary.csv", index=False)
        self.visualizer.plot_duplication_stats(df_dups)

    def run_ortho_kaks_analysis(self):
        """
        Orthologues Ka/Ks Calculation using Biopython PAML.
        """
        # Check if kaks already exists
        kaks_file = self.ortho_root / "ortho_kaks_results.csv"
        if not self.force and kaks_file.exists():
            res_df = pd.read_csv(kaks_file)
            self.visualizer.plot_ortho_kaks_stats(res_df)
            logging.info(f"  [Skip] Orthologues Ka/Ks results found in {kaks_file}")
            return

        orthologues_dir = self.ortho_root / "Orthologues"
        if not orthologues_dir.exists():
            # Try finding in Results_* subdirectory
            found = list(self.ortho_root.glob("Results_*/Orthologues"))
            if found:
                orthologues_dir = found[0]
            else:
                logging.warning("Orthologues directory not found. Please run OrthoFinder first.")
                return

        if self.cds_dir:
            logging.info("Starting Orthologues Ka/Ks analysis...")
        else:
            logging.warning("No CDS directory provided for Orthologues Ka/Ks analysis.")
            return
        
        # We need CDS sequences. Load them.
        cds_files = list(self.cds_dir.glob("*.fa")) + list(self.cds_dir.glob("*.fasta"))
        if not cds_files:
            logging.warning("No CDS files found for Orthologues Ka/Ks analysis.")
            return
            
        # Index CDS
        cds_index = {}
        for f in cds_files:
            cds_index.update(SeqIO.to_dict(SeqIO.parse(f, "fasta")))

        # Load Pan-genome classification to tag results
        # Map Gene -> OGG -> Class
        gene_to_ogg = {}
        ogg_to_class = {}
        
        og_file = self.ortho_root / "Orthogroups" / "Orthogroups.tsv"
        if not og_file.exists():
            found = list(self.ortho_root.glob("Results_*/Orthogroups/Orthogroups.tsv"))
            if found: 
                og_file = found[0]
        
        if og_file.exists():
            og_df = pd.read_csv(og_file, sep='\t')
            genome_cols = [c for c in og_df.columns if c != 'Orthogroup']
            num_genomes = len(genome_cols)
            
            for _, row in og_df.iterrows():
                ogg = row['Orthogroup']
                present_count = 0
                for col in genome_cols:
                    if pd.notna(row[col]):
                        present_count += 1
                        genes = str(row[col]).split(', ')
                        for g in genes:
                            gene_to_ogg[g] = ogg
                
                ratio = present_count / num_genomes
                if ratio == 1.0: cls = "Core"
                elif ratio >= 0.9: cls = "Softcore"
                elif present_count > 2: cls = "Shell"
                elif present_count == 1: cls = "Specific"
                else: cls = "Shell"
                ogg_to_class[ogg] = cls

        # Build ID Mapping (New_ID -> Original_ID)
        new_to_orig = {}
        if hasattr(self, 'all_families_stats') and self.all_families_stats:
             for item in self.all_families_stats:
                 new_to_orig[item['New_ID']] = item['Original_ID']
        else:
             logging.warning("Family stats not found. Assuming CDS IDs match OrthoFinder IDs (Original IDs).")

        tasks = [] # Accumulate tasks for optimization
        results = [] # Store final results
        
        # Iterate over genomes pairs (simplified)
        for genomes_dir in orthologues_dir.iterdir():
            if not genomes_dir.is_dir(): continue
            
            for tsv in genomes_dir.glob("*.tsv"):
                pairs_df = pd.read_csv(tsv, sep='\t')
                cols = pairs_df.columns
                if len(cols) < 3: continue
                sp1, sp2 = cols[1], cols[2]
                
                for _, row in pairs_df.iterrows():
                    g1_list = str(row[sp1]).split(', ')
                    g2_list = str(row[sp2]).split(', ')
                    
                    if len(g1_list) == 1 and len(g2_list) == 1:
                        g1, g2 = g1_list[0], g2_list[0]
                        
                        # Map to Original IDs for CDS lookup
                        g1_lookup = new_to_orig.get(g1, g1)
                        g2_lookup = new_to_orig.get(g2, g2)
                        
                        if g1_lookup in cds_index and g2_lookup in cds_index:
                            ogg = gene_to_ogg.get(g1, "Unknown")
                            cls = ogg_to_class.get(ogg, "Unknown")
                            map_info = {'Gene1': g1, 'Gene2': g2, 'OGG': ogg, 'Class': cls}
                            tasks.append((cds_index[g1_lookup], cds_index[g2_lookup], map_info))

        # Parallel Execution using multiprocessing.Pool (faster than ProcessPoolExecutor)
        logging.info(f"Running Ka/Ks calculation for {len(tasks)} pairs using {self.threads} threads...")
        
        if tasks:
            batch_tasks = []
            for t in tasks:
                # Pack args: (seq1, seq2) - no tmp_dir needed
                batch_tasks.append((t[0], t[1]))
            
            from multiprocessing import Pool
            with Pool(processes=int(self.threads)) as pool:
                # Use imap_unordered with chunksize for better performance
                future_to_info = {i: tasks[i][2] for i in range(len(batch_tasks))}
                iterator = tqdm(pool.imap_unordered(calculate_homo_kaks, batch_tasks, chunksize=20),
                              total=len(batch_tasks), desc="Calculating Ka/Ks", unit="pair")

                for idx, res in enumerate(iterator):
                    info = future_to_info[idx]  # Retrieve info for this task index
                    if res:
                        # Merge logical info (OGG, Class) with result
                        res['OGG'] = info['OGG']
                        res['Class'] = info['Class']
                        res['Gene1'] = info['Gene1']
                        res['Gene2'] = info['Gene2']
                        results.append(res)
        
        if results:
            res_df = pd.DataFrame(results)
            res_df.to_csv(self.ortho_root / "ortho_kaks_results.csv", index=False)
            self.visualizer.plot_ortho_kaks_stats(res_df)

    def run_univariate_association(self):
        """
        Univariate OGG-phenotype association analysis with separate CNV and PAV handling.
        Core OGGs (100% presence) are analyzed using CNV (copy number) with Spearman correlation.
        Non-core OGGs (Softcore/Shell/Specific) are analyzed using PAV (presence/absence) with Point-Biserial.
        """
        if not self.phe_file:
            logging.info("  [Skip] No phenotype file provided for association analysis.")
            return

        logging.info("Starting Univariate OGG-Phenotype Association Analysis (CNV + PAV)...")

        # 1. Load Phenotype Data
        try:
            pheno_df = pd.read_csv(self.phe_file)
            if pheno_df.shape[1] < 2:
                logging.warning("Phenotype file must have at least 2 columns (Genome, Trait).")
                return
            pheno_df.columns = ['Genome'] + list(pheno_df.columns[1:])
            pheno_df['Genome'] = pheno_df['Genome'].astype(str)
            pheno_df.set_index('Genome', inplace=True)
        except Exception as e:
            logging.error(f"Failed to read phenotype file: {e}")
            return

        # 2. Load Gene Count Matrix
        og_counts_file = self.ortho_root / "Orthogroups" / "Orthogroups.GeneCount.tsv"
        if not og_counts_file.exists():
            found = list(self.ortho_root.glob("Results_*/Orthogroups/Orthogroups.GeneCount.tsv"))
            if found:
                og_counts_file = found[0]
            else:
                logging.warning("Orthogroups.GeneCount.tsv not found but required for association analysis.")
                return

        counts_df = pd.read_csv(og_counts_file, sep='\t')
        
        # 3. Load Pan-Genome Classification
        class_file = self.ortho_root / "pan_genome_classification.csv"
        ogg_class_map = {}
        if class_file.exists():
            try:
                class_df = pd.read_csv(class_file)
                if 'Orthogroup' in class_df.columns and 'Class' in class_df.columns:
                    ogg_class_map = dict(zip(class_df['Orthogroup'], class_df['Class']))
                    logging.info(f"Loaded pan-genome classification for {len(ogg_class_map)} OGGs.")
            except Exception as e:
                logging.warning(f"Could not load classification file: {e}")
        else:
            logging.warning("Pan-genome classification file not found. All OGGs will be treated as PAV.")
        
        # 4. Separate OGGs into Core (CNV) and Non-Core (PAV)
        genome_cols = [c for c in counts_df.columns if c not in ['Orthogroup', 'Total']]
        counts_df.set_index('Orthogroup', inplace=True)
        
        core_oggs = [ogg for ogg in counts_df.index if ogg_class_map.get(ogg) == 'Core']
        noncore_oggs = [ogg for ogg in counts_df.index if ogg_class_map.get(ogg) in ['Softcore', 'Shell', 'Specific']]
        
        # If no classification, treat all as PAV
        if not ogg_class_map:
            noncore_oggs = list(counts_df.index)
            core_oggs = []
        
        logging.info(f"  Core OGGs (CNV analysis): {len(core_oggs)}")
        logging.info(f"  Non-Core OGGs (PAV analysis): {len(noncore_oggs)}")
        
        # 5. Prepare Matrices
        # CNV Matrix: Keep original counts, apply cap if configured
        if core_oggs and self.cnv_assoc:
            X_cnv = counts_df.loc[core_oggs, genome_cols].T
            if self.cnv_cap is not None:
                X_cnv = X_cnv.clip(upper=self.cnv_cap)
                logging.info(f"  Applied CNV cap at {self.cnv_cap}")
            X_cnv.index = X_cnv.index.astype(str)
        else:
            X_cnv = pd.DataFrame()
        
        # PAV Matrix: Binarize counts (>0 -> 1)
        if noncore_oggs and self.pav_assoc:
            X_pav = (counts_df.loc[noncore_oggs, genome_cols] > 0).astype(int).T
            X_pav.index = X_pav.index.astype(str)
        else:
            X_pav = pd.DataFrame()
        
        # 6. Align Data with Phenotypes
        common_genomes = pheno_df.index.intersection(genome_cols)
        if len(common_genomes) < 5:
            logging.warning(f"Only {len(common_genomes)} samples overlap. Need at least 5. Skipping.")
            return
        
        logging.info(f"Aligned Dataset: {len(common_genomes)} samples.")
        
        # Import statistical libraries
        from scipy.stats import pointbiserialr, spearmanr, kendalltau
        from statsmodels.stats.multitest import multipletests

        # Collect results
        cnv_results = []
        pav_results = []
        
        # 7. Analyze Each Trait
        for trait in pheno_df.columns:
            y_all = pheno_df.loc[common_genomes, trait]
            
            # Remove NaN
            valid_mask = ~y_all.isna()
            if valid_mask.sum() < 5:
                logging.warning(f"Skipping trait {trait}: <5 valid samples.")
                continue
            
            y_trait = y_all.loc[valid_mask]
            logging.info(f"  Analyzing trait: {trait} (n={len(y_trait)})")
            
            # --- CNV Analysis (Spearman/Kendall) ---
            if not X_cnv.empty and self.cnv_assoc:
                # Restrict to common_genomes first, then apply valid_mask
                X_cnv_common = X_cnv.loc[common_genomes]
                X_cnv_trait = X_cnv_common.loc[valid_mask]
                logging.info(f"    CNV analysis: {X_cnv_trait.shape[1]} core OGGs using {self.cnv_method}")
                
                for ogg in X_cnv_trait.columns:
                    x_ogg = X_cnv_trait[ogg].values
                    y_vals = y_trait.values
                    
                    # Check variance
                    if x_ogg.std() == 0 or y_vals.std() == 0:
                        continue
                    
                    try:
                        if self.cnv_method == 'spearman':
                            corr, pval = spearmanr(x_ogg, y_vals)
                        elif self.cnv_method == 'kendall':
                            corr, pval = kendalltau(x_ogg, y_vals)
                        else:
                            corr, pval = spearmanr(x_ogg, y_vals)  # Default
                        
                        cnv_results.append({
                            'Trait': trait,
                            'Orthogroup': ogg,
                            'Effect_Size': corr,
                            'P_value': pval,
                            'Matrix_Type': 'CNV',
                            'Class': 'Core'
                        })
                    except Exception:
                        continue
            
            # --- PAV Analysis (Point-Biserial) ---
            if not X_pav.empty and self.pav_assoc:
                # Restrict to common_genomes first, then apply valid_mask
                X_pav_common = X_pav.loc[common_genomes]
                X_pav_trait = X_pav_common.loc[valid_mask]
                logging.info(f"    PAV analysis: {X_pav_trait.shape[1]} non-core OGGs using {self.pav_method}")
                
                for ogg in X_pav_trait.columns:
                    x_ogg = X_pav_trait[ogg].values
                    y_vals = y_trait.values
                    
                    # Check variance
                    if x_ogg.std() == 0 or y_vals.std() == 0:
                        continue
                    
                    try:
                        corr, pval = pointbiserialr(x_ogg, y_vals)
                        
                        pav_results.append({
                            'Trait': trait,
                            'Orthogroup': ogg,
                            'Effect_Size': corr,
                            'P_value': pval,
                            'Matrix_Type': 'PAV',
                            'Class': ogg_class_map.get(ogg, 'Unknown')
                        })
                    except Exception:
                        continue
        
        # 8. FDR Correction and Save Results
        # Separate FDR correction for CNV and PAV
        if cnv_results:
            cnv_df = pd.DataFrame(cnv_results)
            # Per-trait FDR correction
            cnv_df['FDR'] = 1.0
            for trait in cnv_df['Trait'].unique():
                mask = cnv_df['Trait'] == trait
                pvals = cnv_df.loc[mask, 'P_value']
                try:
                    _, fdr_vals, _, _ = multipletests(pvals, method='fdr_bh')
                    cnv_df.loc[mask, 'FDR'] = fdr_vals
                except Exception:
                    pass
            
            cnv_df = cnv_df.sort_values(['Trait', 'P_value'])
            cnv_out = self.assoc_root / "cnv_association_results.csv"
            cnv_df.to_csv(cnv_out, index=False)
            logging.info(f"Saved CNV results to {cnv_out} ({len(cnv_df)} associations)")
        
        if pav_results:
            pav_df = pd.DataFrame(pav_results)
            # Per-trait FDR correction
            pav_df['FDR'] = 1.0
            for trait in pav_df['Trait'].unique():
                mask = pav_df['Trait'] == trait
                pvals = pav_df.loc[mask, 'P_value']
                try:
                    _, fdr_vals, _, _ = multipletests(pvals, method='fdr_bh')
                    pav_df.loc[mask, 'FDR'] = fdr_vals
                except Exception:
                    pass
            
            pav_df = pav_df.sort_values(['Trait', 'P_value'])
            pav_out = self.assoc_root / "pav_association_results.csv"
            pav_df.to_csv(pav_out, index=False)
            logging.info(f"Saved PAV results to {pav_out} ({len(pav_df)} associations)")
        
        # 9. Save Combined Results
        if cnv_results or pav_results:
            all_res = cnv_results + pav_results
            combined_df = pd.DataFrame(all_res).sort_values(['Trait', 'P_value'])
            combined_out = self.assoc_root / "combined_association_results.csv"
            combined_df.to_csv(combined_out, index=False)
            logging.info(f"Saved combined results to {combined_out}")
            
            # 10. Generate Visualizations
            logging.info("Generating visualizations...")
            
            for trait in combined_df['Trait'].unique():
                trait_df = combined_df[combined_df['Trait'] == trait].copy()
                
                # Volcano plot (distinguish CNV vs PAV)
                self.visualizer.plot_volcano(trait_df, trait)
                
                # Get significant results
                sig_results = trait_df[trait_df['P_value'] < 0.05].copy()
                
                if not sig_results.empty:
                    # PAV boxplots for significant PAV results
                    pav_sig = sig_results[sig_results['Matrix_Type'] == 'PAV']
                    if not pav_sig.empty:
                        top_pav = pav_sig.nsmallest(min(len(pav_sig), 12), 'P_value')
                        pav_oggs = top_pav['Orthogroup'].tolist()
                        X_pav_vis = X_pav.loc[common_genomes, [o for o in pav_oggs if o in X_pav.columns]]
                        self.visualizer.plot_pav_boxplot(X_pav_vis, pheno_df, top_pav, trait)
                    
                    # CNV scatter plots for significant CNV results
                    cnv_sig = sig_results[sig_results['Matrix_Type'] == 'CNV']
                    if not cnv_sig.empty:
                        top_cnv = cnv_sig.nsmallest(min(len(cnv_sig), 9), 'P_value')
                        cnv_oggs = top_cnv['Orthogroup'].tolist()
                        X_cnv_vis = X_cnv.loc[common_genomes, [o for o in cnv_oggs if o in X_cnv.columns]]
                        self.visualizer.plot_cnv_scatter(X_cnv_vis, pheno_df, top_cnv, trait)
        else:
            logging.warning("No significant associations found for any trait.")

    def run_expression_analysis(self):
        """
        Expression Analysis.
        Reads config[genome][name]['expression'] and ['group'].
        """
        genome_configs = self.config.get('genome', {})
        if not genome_configs:
            return

        logging.info("Starting Expression analysis...")
        # Build Global Mapping (Original ID -> New ID)
        mapping_db = {}
        id_fam_map = {}
        for record in self.all_families_stats:
            mapping_db[str(record['Original_ID'])] = record['New_ID']
            id_fam_map[record['New_ID']] = record['Family']

        for gname, gcfg in genome_configs.items():
            exp_file = gcfg.get('expression')
            if not exp_file: 
                continue
            
            exp_path = Path(exp_file)
            if not exp_path.exists():
                logging.warning(f"Expression file not found for {gname}: {exp_path}")
                continue

            logging.info(f"  Analyzing expression for {gname}...")
            
            try:
                # Load Expression
                exp_df = pd.read_csv(exp_path, sep='\t', index_col=0)
                logging.info(f"    Loaded expression data: {exp_df.shape[0]} genes x {exp_df.shape[1]} samples.")
                
                # Load Groups (Optional)
                group_map = {}
                group_file = gcfg.get('group')
                # Determine separator
                sep = ','
                if group_file.endswith('.tsv') or group_file.endswith('.txt'):
                    sep = '\t'
                if group_file and Path(group_file).exists():
                    gdf = pd.read_csv(group_file, sep=sep, header=0)
                    # Assume headers Sample, Group or col 0, col 1
                    if len(gdf.columns) >= 2:
                        for _, row in gdf.iterrows():
                            group_map[str(row.iloc[0])] = str(row.iloc[1])
                logging.info(f"    Loaded {len(group_map)} sample groups.")

                # Process
                plot_data = []
                heatmap_data_list = []
                
                for orig_id, row_vals in exp_df.iterrows():
                    orig_id_str = str(orig_id)
                    if orig_id_str not in mapping_db: continue
                    
                    new_id = mapping_db[orig_id_str]
                    fam = id_fam_map.get(new_id, "Unknown")
                    # Try getting class from mapped ID first, if not try original
                    cls = self.gene_class_map.get(new_id, self.gene_class_map.get(orig_id_str, "Unknown"))

                    vals = pd.to_numeric(row_vals, errors='coerce').fillna(0)
                    mean_exp = vals.mean()
                    log_tpm = np.log2(mean_exp + 1)
                    
                    # Store 
                    plot_data.append({
                        'ID': new_id,
                        'Family': fam,
                        'Class': cls,
                        'Exp_Mean': log_tpm
                    })
                    
                    # Compute Tau if groups exist
                    if group_map:
                        # Subset columns that are in group_map
                        valid_cols = [c for c in vals.index if str(c) in group_map]
                        if valid_cols:
                            sub_vals = vals[valid_cols]
                            # specificty calculation (simple Tau)
                            # x_hat = x / max(x)
                            # tau = sum(1 - x_hat) / (N - 1)
                            # Group by tissue first
                            grp_means = sub_vals.groupby(lambda x: group_map[str(x)]).mean()
                            if grp_means.max() > 0:
                                x_hat = grp_means / grp_means.max()
                                tau = (1 - x_hat).sum() / (len(grp_means) - 1)
                                plot_data[-1]['Tau'] = tau
                            
                            # Collect data for heatmap: {Gene: {Tissue: LogExp}}
                            log_grp_means = np.log2(grp_means + 1)
                            hm_data = log_grp_means.to_dict()
                            hm_data['ID'] = new_id
                            heatmap_data_list.append(hm_data)

                # Plot Stats (Tau)
                self.visualizer.plot_expression_stats(plot_data, gname)
                
                # Plot Heatmap
                if heatmap_data_list and len(plot_data) > 0:
                    # 1. Create Heatmap DF (Index=Gene, Cols=Tissues)
                    hm_df = pd.DataFrame(heatmap_data_list)
                    if 'ID' in hm_df.columns:
                        hm_df.set_index('ID', inplace=True)
                    
                    # 2. Create Meta DF (Index=Gene, Cols=Class, Tau)
                    meta_df = pd.DataFrame(plot_data)
                    if 'ID' in meta_df.columns:
                        meta_df.set_index('ID', inplace=True)
                    
                    # Ensure we have Tau and Class
                    if 'Tau' in meta_df.columns and 'Class' in meta_df.columns:
                        self.visualizer.plot_expression_heatmap(hm_df, meta_df, gname)
                
            except Exception as e:
                logging.error(f"Expression analysis failed for {gname}: {e}")

    def run_te_analysis(self):
        """
        Phase: TE Density Analysis.
        Calculates TE coverage in upstream (2kb), gene body, and downstream (2kb) regions.
        Performs analysis for each genome defined in config.
        """
        genome_configs = self.config.get('genome', {})
        if not genome_configs:
            return

        logging.info("Starting TE Density analysis...")
        # Build Global Mapping (Original ID -> New ID)
        mapping_db = {}
        if hasattr(self, 'all_families_stats'):
            for record in self.all_families_stats:
                mapping_db[str(record['Original_ID'])] = record['New_ID']
        
        for gname, gcfg in genome_configs.items():
            gene_file = gcfg.get('gene_bed')
            te_file = gcfg.get('te_bed')
            
            if not gene_file or not te_file:
                continue

            gene_path = Path(gene_file)
            te_path = Path(te_file)
            
            if not gene_path.exists():
                logging.error(f"  [Error] {gname}: Gene file not found: {gene_path}")
                continue
            if not te_path.exists():
                logging.error(f"  [Error] {gname}: TE file not found: {te_path}")
                continue

            logging.info(f"  Analyzing TE profile for {gname}...")
            logging.info(f"  Loading annotations: Genes: {gene_path}, TEs: {te_path}")

            # 1. Load Genes (BED)
            try:
                df_g = pd.read_csv(gene_path, sep='\t', comment='#', header=None)
                if df_g.shape[1] >= 6: # BED6 required for strand
                    df_g = df_g.iloc[:, [0, 1, 2, 3, 5]]
                    df_g.columns = ['chrom', 'start', 'end', 'ID', 'strand']
                    logging.info(f"    Loaded {len(df_g)} genes.")
                else:
                    logging.error(f"  [Error] {gname}: Unsupported gene file format. Please use BED6.")
                    continue
            except Exception as e:
                logging.error(f"  [Error] {gname}: Loading gene file: {e}")
                continue
                
            # Filtering: Keep only genes in the analysis (Family members)
            # & Map Original IDs to Class
            valid_indices = []
            gene_classes = []
            
            for idx, row in df_g.iterrows():
                orig_id = str(row['ID'])
                new_id = mapping_db.get(orig_id) # Try map Original -> New
                
                cls = None
                if new_id:
                    cls = self.gene_class_map.get(new_id)
                else:
                    # Fallback: maybe gene_class_map has original IDs or IDs match
                    cls = self.gene_class_map.get(orig_id)
                
                if cls and cls in ["Core", "Softcore", "Shell", "Specific"]:
                    valid_indices.append(idx)
                    gene_classes.append(cls)
            
            if not valid_indices:
                logging.warning(f"  [Warning] {gname}: No family members found in gene file. Skipping TE analysis.")
                continue

            df_g = df_g.loc[valid_indices].copy()
            df_g['Class'] = gene_classes
            logging.info(f"    Filtered to {len(df_g)} family member genes.")

            # 2. Load TEs (BED)
            try:
                df_t = pd.read_csv(te_path, sep='\t', comment='#', header=None)
                if df_t.shape[1] >= 3:
                    df_t = df_t.iloc[:, [0, 1, 2]]
                    df_t.columns = ['chrom', 'start', 'end']
                    logging.info(f"    Loaded {len(df_t)} TEs.")
                else:
                    logging.error(f"  [Error] {gname}: Unsupported TE file format. Please use BED3.")
                    continue

                df_t['start'] = pd.to_numeric(df_t['start'], errors='coerce')
                df_t['end'] = pd.to_numeric(df_t['end'], errors='coerce')
                df_t.dropna(subset=['start', 'end'], inplace=True)
            except Exception as e:
                logging.error(f"  [Error] {gname}: Loading TE file: {e}")
                continue

            # 3. Analyze Coverage
            profile_sums = defaultdict(lambda: np.zeros(90))
            class_counts = defaultdict(int)
            
            # Helper: group TEs
            te_groups = {k: v.sort_values('start') for k, v in df_t.groupby('chrom')}
            
            for chrom, group in df_g.groupby('chrom'):
                if chrom not in te_groups: continue
                
                te_data = te_groups[chrom]
                te_starts = te_data['start'].values
                te_ends = te_data['end'].values
                
                for _, row in group.iterrows():
                    g_cls = row['Class'] # Use pre-calculated Class
                    if not g_cls: continue
                    
                    class_counts[g_cls] += 1
                    g_s, g_e = int(row['start']), int(row['end'])
                    strand = row['strand']
                    
                    # ROI: Gene +/- 2kb
                    roi_s, roi_e = g_s - 2000, g_e + 2000
                    
                    # Filter relevant TEs
                    idx_end = np.searchsorted(te_starts, roi_e, side='left')
                    idx_start = np.searchsorted(te_starts, roi_s - 50000, side='left')
                    idx_start = max(0, idx_start)
                    
                    c_starts = te_starts[idx_start:idx_end]
                    c_ends = te_ends[idx_start:idx_end]
                    
                    mask = c_ends > roi_s
                    rel_s = c_starts[mask]
                    rel_e = c_ends[mask]
                    
                    if len(rel_s) == 0: continue
                    
                    # Meta-gene Binning method
                    # Map to 90 bins (30 Upstream, 30 Body, 30 Downstream)
                    bins = []
                    bins.extend(zip(np.linspace(g_s - 2000, g_s, 31)[:-1], np.linspace(g_s - 2000, g_s, 31)[1:]))
                    bins.extend(zip(np.linspace(g_s, g_e, 31)[:-1], np.linspace(g_s, g_e, 31)[1:]))
                    bins.extend(zip(np.linspace(g_e, g_e + 2000, 31)[:-1], np.linspace(g_e, g_e + 2000, 31)[1:]))
                    
                    gene_profile = []
                    for b_s, b_e in bins:
                        b_len = b_e - b_s
                        if b_len <= 0: 
                            gene_profile.append(0)
                            continue
                        
                        ov_s = np.maximum(rel_s, b_s)
                        ov_e = np.minimum(rel_e, b_e)
                        ov_len = np.maximum(0, ov_e - ov_s)
                        
                        # Calculation Metric: Insertion Density (Events per kb)
                        # Count how many TE fragments overlap with this bin
                        count = np.sum(ov_len > 0)
                        
                        # Normalize by bin width (in kb) to get density
                        bin_size_kb = b_len / 1000.0
                        density = count / bin_size_kb if bin_size_kb > 0 else 0
                        
                        gene_profile.append(density)
                    
                    arr = np.array(gene_profile)
                    if strand == '-':
                        arr = arr[::-1]
                    
                    profile_sums[g_cls] += arr

            # 4. Normalize & Plot
            final_data = {}
            for cls, count in class_counts.items():
                if count > 0:
                    final_data[cls] = profile_sums[cls] / count
            
            if final_data:
                self.visualizer.plot_te_profile(final_data, gname)
            else:
                logging.warning(f"  [Warning] {gname}: No TE profile data generated.")

    def run_hmm_search(self, hmm_path, proteome_path, output_path, param_dict=None):
        """Run hmmsearch, return parsed hits."""
        
        # Determine HMM files
        hmm_files = []
        if isinstance(hmm_path, list):
            hmm_files = [Path(p) for p in hmm_path]
        else:
            hmm_files = [Path(hmm_path)]
            
        # Determine parameters
        evalue = self.hmm_evalue
        if param_dict:
             # Priority: specific 'hmm_evalue' > global self.hmm_evalue
             evalue = param_dict.get('hmm_evalue', evalue)

        all_hits = []
        
        for i, h_file in enumerate(hmm_files):
            # Define output path
            if len(hmm_files) > 1:
                # Insert index before suffix
                out_p = Path(output_path)
                out_file = out_p.parent / f"{out_p.stem}.{i+1}{out_p.suffix}"
            else:
                out_file = Path(output_path)
            
            if not self.force and out_file.exists() and out_file.stat().st_size > 0:
                logging.info(f"    [Skip] HMM search {i+1}/{len(hmm_files)} (Result exists)")
                hits = self.parse_domtbl_with_domains(out_file, param_dict)
            else:
                logging.info(f"    Running HMM search {i+1}/{len(hmm_files)}...")
                cmd = f"hmmsearch --cpu {self.threads} -E {evalue} --domtblout {out_file} {h_file} {proteome_path}"
                self.run_command(cmd)
                hits = self.parse_domtbl_with_domains(out_file, param_dict)
            
            logging.info(f"    -> Found {len(hits)} hits from {h_file.name}")
            all_hits.append(hits)
        
        # Merge Results (Strict AND Logic)
        if not all_hits:
            return {}
            
        merged_hits = defaultdict(list)
        
        # AND: Intersection of genes
        if all_hits:
            common_genes = set(all_hits[0].keys())
            for h in all_hits[1:]:
                common_genes.intersection_update(h.keys())
            
            for gene in common_genes:
                for h in all_hits:
                    if gene in h:
                        merged_hits[gene].extend(h[gene])
                        
        return dict(merged_hits)

    def run_blast_search(self, query_fasta, proteome_path, output_path, param_dict):
        """
        Run diamond blastp.
        param_dict contains evalue, identity, coverage etc.
        """
        if not self.force and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            logging.info(f"  [Skip] BLAST search (Result exists)")
            hits = self.parse_blast_output(output_path)
            logging.info(f"    -> Found {len(hits)} hits from {Path(query_fasta).name}")
            return hits

        db_prefix = str(proteome_path)
        if not Path(f"{db_prefix}.dmnd").exists():
            self.run_command(f"diamond makedb --in {proteome_path} --db {db_prefix} --quiet")
        
        blast_conf = param_dict.get('blast', {})
        
        # Evalue
        evalue = blast_conf.get('blast_evalue', blast_conf.get('evalue', param_dict.get('blast_evalue', self.blast_evalue)))
        
        # Identity
        ident = blast_conf.get('blast_identity', blast_conf.get('identity', blast_conf.get('pident', param_dict.get('blast_identity', self.blast_identity))))
        
        # Coverage
        cov = blast_conf.get('blast_coverage', blast_conf.get('coverage', blast_conf.get('qcov', param_dict.get('blast_coverage', self.blast_coverage))))
        
        args = []
        if float(ident) > 0: args.append(f"--id {ident}")
        if float(cov) > 0: args.append(f"--query-cover {cov}")
        
        logging.info(f"    Running BLAST search...")
        cmd = f"diamond blastp --quiet --threads {self.threads} --query {query_fasta} --db {db_prefix} --out {output_path} --outfmt 6 --evalue {evalue} --max-target-seqs 10000 {' '.join(args)}"
        self.run_command(cmd)
        hits = self.parse_blast_output(output_path)
        logging.info(f"    -> Found {len(hits)} hits from {Path(query_fasta).name}")
        return hits

    def run_hmm_blast_search(self, hmm_path, query_fasta, proteome_path, fam_dir, genome_name, param_dict):
        """Run HMM, extract hits, then BLAST them against query_fasta."""
        hmm_out = fam_dir / f"{genome_name}.hmm.domtbl"
        hmm_hits = self.run_hmm_search(hmm_path, proteome_path, hmm_out, param_dict)
        
        if not hmm_hits: return {}
        
        tmp_hits_fa = fam_dir / f"{genome_name}.hmm_candidates.fa"
        
        prot_dict = SeqIO.index(str(proteome_path), "fasta")
        count = 0
        with open(tmp_hits_fa, "w") as f:
            for gid in hmm_hits:
                if gid in prot_dict:
                    f.write(f">{gid}\n{prot_dict[gid].seq}\n")
                    count += 1
        if count == 0: return {}
        
        seed_db = str(query_fasta)
        if not Path(f"{seed_db}.dmnd").exists():
            self.run_command(f"diamond makedb --in {query_fasta} --db {seed_db} --quiet")
            
        blast_out = fam_dir / f"{genome_name}.hmm_blast.tsv"
        
        # Use BLAST config for filtering
        blast_conf = param_dict.get('blast', {})
        evalue = blast_conf.get('blast_evalue', blast_conf.get('evalue', param_dict.get('blast_evalue', self.blast_evalue)))
        ident = blast_conf.get('blast_identity', blast_conf.get('identity', blast_conf.get('pident', param_dict.get('blast_identity', self.blast_identity))))
        cov = blast_conf.get('blast_coverage', blast_conf.get('coverage', blast_conf.get('qcov', param_dict.get('blast_coverage', self.blast_coverage))))
        
        args = []
        if float(ident) > 0: args.append(f"--id {ident}")
        if float(cov) > 0: args.append(f"--query-cover {cov}")

        logging.info(f"    Running filtering BLAST step...")
        cmd = f"diamond blastp --quiet --threads {self.threads} --query {tmp_hits_fa} --db {seed_db} --out {blast_out} --outfmt 6 --evalue {evalue} --max-target-seqs 1 {' '.join(args)}"
        self.run_command(cmd)
        
        valid_candidates = set()
        if blast_out.exists():
            with open(blast_out, 'r') as f:
                for line in f:
                    parts = line.split('\t')
                    if len(parts) >= 1:
                        valid_candidates.add(parts[0])
                    
        final_hits = {k: v for k, v in hmm_hits.items() if k in valid_candidates}
        logging.info(f"    -> Retained {len(final_hits)} hits after filtering")
        return final_hits

    def run_blast_hmm_search(self, hmm_path, query_fasta, proteome_path, fam_dir, genome_name, param_dict):
        """Run BLAST, extract hits, then HMM scan them."""
        blast_out = fam_dir / f"{genome_name}.blast.tsv"
        
        # BLAST Step
        blast_hits = self.run_blast_search(query_fasta, proteome_path, blast_out, param_dict)
        
        if not blast_hits: return {}
        
        tmp_hits_fa = fam_dir / f"{genome_name}.blast_candidates.fa"
        prot_dict = SeqIO.index(str(proteome_path), "fasta")
        with open(tmp_hits_fa, "w") as f:
            for gid in blast_hits:
                if gid in prot_dict:
                    f.write(f">{gid}\n{prot_dict[gid].seq}\n")
                    
        hmm_out = fam_dir / f"{genome_name}.blast_hmm.domtbl"
        
        # HMM Step
        # Pass param_dict for HMM config too
        hmm_hits = self.run_hmm_search(hmm_path, str(tmp_hits_fa), hmm_out, param_dict)
        logging.info(f"    -> Retained {len(hmm_hits)} hits after filtering")
        
        return hmm_hits

    def preprocess_proteomes(self):
        """
        Reads raw proteome files, keeps only the longest transcript per gene,
        and saves them to a temporary directory.
        """
        raw_files = self.get_proteome_files()
        clean_dir = self.output_dir / "cleaned_proteomes"
        clean_dir.mkdir(exist_ok=True)
        
        cleaned_files = []
        logging.info("Preprocessing proteomes: Keeping longest transcripts...")
        
        for p_file in raw_files:
            out_file = clean_dir / p_file.name
            
            # Optimization: Skip if exists
            if not self.force and out_file.exists() and out_file.stat().st_size > 0:
                cleaned_files.append(out_file)
                continue
                
            # Logic: Group by gene ID (remove .t1, .1 suffix)
            gene_map = {} # gene_id -> record
            for record in SeqIO.parse(p_file, "fasta"):
                # Determine gene ID
                # Rule: Split by last dot to remove version/transcript suffix
                if "." in record.id:
                    gene_id = record.id.rsplit(".", 1)[0]
                else:
                    gene_id = record.id
                    
                if gene_id not in gene_map:
                    gene_map[gene_id] = record
                else:
                    if len(record.seq) > len(gene_map[gene_id].seq):
                        gene_map[gene_id] = record
            
            if gene_map:
                SeqIO.write(list(gene_map.values()), out_file, "fasta")
                cleaned_files.append(out_file)
            else:
                 logging.warning(f"  {p_file.name} yielded no sequences.")
                 
        return cleaned_files

    def process(self):
        """
        Run the complete analysis pipeline.

        Execution Phases:
        1. Search: Identify gene family members via HMM/BLAST/HMM-BLAST/BLAST-HMM.
        2. Family Analysis: Extract sequences, align, and build trees.
        3. Orthology: Identify orthogroups via OrthoFinder.
        4. Evolution: Pan-genome stats, duplications, and Selection pressure (Ka/Ks).
        5. Characterization: Expression and TE analyses for each genome.
        6. Mining: ML-based or Univariate-based Phenotype Association.
        """
        # --- Step 0: Preprocessing  ---
        proteome_files = self.preprocess_proteomes()
        if not proteome_files:
            logging.error(f"No valid proteome files found in {self.input_dir} (or preprocessing failed)")
            return
        
        logging.info(f"Using {len(proteome_files)} proteome files from {self.output_dir}/cleaned_proteomes")

        # --- Step 1: Family-wise Search & Feature Extraction ---
        for fam_name, hmm_path_input in self.hmm_configs.items():
            logging.info(f"Processing Family: {fam_name}...")
            fam_dir = self.family_root / fam_name
            fam_dir.mkdir(exist_ok=True)
            method = self.search_method
            logging.info(f"  Method: {method}")

            # Prepare files
            final_hmms = []
            final_fasta = None 
            
            # Parse inputs (Comma separated: HMMs and/or FASTA)
            raw_input = str(hmm_path_input)
            parts = [p.strip() for p in raw_input.split(',') if p.strip()]
            
            for p_str in parts:
                path = Path(p_str)
                if not path.exists():
                    logging.warning(f"  Input file {path} not found.")
                    continue
                
                # Distinguish by extension
                if path.suffix.lower() in ['.fa', '.fasta', '.faa', '.pep']:
                    if final_fasta is None:
                        final_fasta = path
                    else:
                        logging.warning(f"  Multiple FASTA files not supported for BLAST seed. Ignoring: {path}")
                else:
                    final_hmms.append(path)

            # Get Search Parameters from Config (e.g. thresholds)
            fam_conf = self.config.get('family', {}).get(fam_name, {})
            if not isinstance(fam_conf, dict): fam_conf = {}


            # Check availability for selected method
            has_hmm = (len(final_hmms) > 0)
            has_fasta = (final_fasta is not None and final_fasta.exists())

            # Validate requirements
            if method == 'hmm' and not has_hmm:
                logging.error(f"  [Error] Method is HMM but HMM file unavailable for {fam_name}. (Check inputs)")
                continue
            if method == 'blast' and not has_fasta:
                logging.error(f"  [Error] Method is BLAST but FASTA file unavailable for {fam_name}. (Check inputs)")
                continue
            if method in ['hmm_blast', 'blast_hmm'] and not (has_hmm and has_fasta):
                logging.error(f"  [Error] Method is {method} but HMM or FASTA file missing for {fam_name}.")
                continue

            current_fam_sequences = []
            current_fam_stats = []

            for proteome_path in proteome_files:
                genome_name = proteome_path.stem
                logging.info(f"  Searching {genome_name}...")
                hits_data = {}
                param_dict = fam_conf
                
                try:
                    if method == 'hmm':
                        hits_data = self.run_hmm_search(final_hmms, str(proteome_path), fam_dir / f"{genome_name}_{fam_name}.domtbl", param_dict)
                    
                    elif method == 'blast':
                        hits_data = self.run_blast_search(str(final_fasta), str(proteome_path), fam_dir / f"{genome_name}_{fam_name}.blast", param_dict)

                    elif method == 'hmm_blast': # HMM then BLAST
                        hits_data = self.run_hmm_blast_search(final_hmms, str(final_fasta), str(proteome_path), fam_dir, genome_name, param_dict)

                    elif method == 'blast_hmm': # BLAST then HMM
                        hits_data = self.run_blast_hmm_search(final_hmms, str(final_fasta), str(proteome_path), fam_dir, genome_name, param_dict)
                    else:
                        logging.error(f"  [Error] Unknown method: {method}")
                except Exception as e:
                    logging.error(f"  Failed search for {genome_name}: {e}")
                    hits_data = {}

                count = len(hits_data)
                self.summary_counts[(genome_name, fam_name)] = count
                
                if count == 0:
                    continue

                # --- Extraction ---
                out_fa = fam_dir / f"{genome_name}_members.fa"
                out_stats_tsv = fam_dir / f"{genome_name}_stats.tsv"
                
                genomes_seqs_obj = []
                genomes_info_list = []

                if not self.force and out_fa.exists() and out_stats_tsv.exists() and out_fa.stat().st_size > 0:
                    logging.info(f"  [Skip] Extraction for {genome_name} (Files exist)")
                    genomes_seqs_obj = list(SeqIO.parse(out_fa, "fasta"))
                    genomes_info_list = pd.read_csv(out_stats_tsv, sep='\t').to_dict('records')
                else:
                    try:
                        genomes_seqs_obj, _, genomes_info_list = self.extract_and_rename(
                            proteome_path, hits_data, genome_name, fam_name
                        )
                        SeqIO.write(genomes_seqs_obj, out_fa, "fasta")
                        pd.DataFrame(genomes_info_list).to_csv(out_stats_tsv, sep='\t', index=False)
                    except Exception as e:
                        logging.error(f"  Extraction failed for {genome_name}: {e}")
                        continue
                
                current_fam_sequences.extend(genomes_seqs_obj)
                current_fam_stats.extend(genomes_info_list)
                self.genome_merged_sequences[genome_name].extend(genomes_seqs_obj)

            # Aggregate Family Results
            if current_fam_sequences:
                logging.info(f"  Identified {len(current_fam_sequences)} sequences in total for family {fam_name}")
                total_fasta_path = fam_dir / f"all_{fam_name}_members.fa"
                total_stats_path = fam_dir / f"all_{fam_name}_stats.tsv"
                SeqIO.write(current_fam_sequences, total_fasta_path, "fasta")
                pd.DataFrame(current_fam_stats).to_csv(total_stats_path, sep='\t', index=False)
                
                # Check features config
                features_conf = self.config.get('family', {}).get('features', {})
                enable_tree = features_conf.get('tree', False)
                enable_meme = features_conf.get('meme', False)

                # Build tree per family
                if enable_tree:
                    tree_outfile = fam_dir / f"{fam_name}_family.tree"
                    if not self.force and tree_outfile.exists() and tree_outfile.stat().st_size > 0:
                        logging.info(f"  [Skip] Phylogeny tree for {fam_name} (Result exists)")
                    else:
                        logging.info(f"  Building phylogeny tree for {fam_name}...")
                        self.build_phylogeny(total_fasta_path, fam_dir, f"{fam_name}_family")
                
                # Run MEME (Motif Analysis)
                if enable_meme:
                    meme_out_dir = fam_dir / "meme"
                    if not self.force and (meme_out_dir / "meme.txt").exists():
                        logging.info(f"  [Skip] MEME motif analysis for {fam_name} (Result exists)")
                    else:
                        logging.info(f"  Running MEME motif analysis for {fam_name}...")
                        self.run_meme(total_fasta_path, meme_out_dir)

                self.all_families_sequences.extend(current_fam_sequences)
                self.all_families_stats.extend(current_fam_stats)
            else:
                logging.warning(f"  No sequences found for family {fam_name} across all genomes.")

        # --- Step 2: Family Summary and Super-tree ---
        if self.all_families_sequences:
            logging.info("Finalizing Global Results across all families...")
            super_fasta_path = self.output_dir / "all_family_members.fa"
            super_stats_path = self.output_dir / "all_family_stats.tsv"
            SeqIO.write(self.all_families_sequences, super_fasta_path, "fasta")
            pd.DataFrame(self.all_families_stats).to_csv(super_stats_path, sep='\t', index=False)
            
            self.write_family_counts()
            
            # Build Super Tree for all families
            super_tree_out = self.output_dir / "super_tree.tree"
            if not self.force and super_tree_out.exists() and super_tree_out.stat().st_size > 0:
                logging.info(f"  [Skip] Super-tree construction (Result exists)")
            else:
                logging.info(f"  Building super-tree from {len(self.all_families_sequences)} total sequences...")
                self.build_phylogeny(super_fasta_path, self.output_dir, "super_tree")

        # --- Step 3: Prepare for Orthology Inference ---
        logging.info("Saving genome-wise merged sequences into 'genomes/' directory...")
        for genome_name, seq_records in self.genome_merged_sequences.items():
            SeqIO.write(seq_records, self.genome_root / f"{genome_name}.fa", "fasta")

        # --- Step 4: Orthology Detection (OrthoFinder) ---
        self.run_orthofinder()

        # --- Step 5: Evolutionary Analysis (Pan-genome, Duplication, Ka/Ks) ---
        self.run_pan_genome()
        
        self.run_duplication_analysis()

        self.run_ortho_kaks_analysis()

        # --- Step 6: TE Density Analysis ---
        self.run_te_analysis()

        # --- Step 7: Expression Analysis ---
        self.run_expression_analysis()

        # --- Step 8: Phenotype Association ---
        self.run_univariate_association()

        logging.info(f"Analysis complete. Results organized in {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pan-gene family analysis toolkit: Automated analysis of gene families across genomes")
    parser.add_argument("-i", "--input", required=True, help="Input proteins directory (one file per genome)")
    parser.add_argument("-o", "--output", required=True, help="Main output directory")
    parser.add_argument("-f", "--family", nargs=2, action='append', metavar=('NAME', 'FILE'), 
                        help="Family name and HMM/Fasta file (e.g., -f MYB myb.hmm or -f MYB myb_seeds.fa or -f MYB myb.hmm,myb2.hmm)")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads for analysis (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    parser.add_argument("--config", "-c", help="Path to TOML configuration file for visualization and analysis")
    parser.add_argument("--method", choices=['hmm', 'blast', 'hmm_blast', 'blast_hmm'], default='hmm',
                        help="Search method: hmm, blast, hmm_blast, or blast_hmm (default: hmm)")

    # phylogenetic Tree argument
    parser.add_argument("--tree", help="Path to phylogenetic tree (Newick format)", default=None)

    # Ka/Ks arguments
    parser.add_argument("--cds", help="Directory containing CDS sequences (required for Ka/Ks)")

    # Duplication Analysis arguments
    parser.add_argument("--loc", help="Directory containing gene location files (BED6 format)")

    # Phenotype association arguments
    parser.add_argument("--phe", help="Phenotype data CSV (Genome,Trait)")

    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(args.output).parent / "gfat.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Suppress verbose fontTools logs
    logging.getLogger('fontTools').setLevel(logging.WARNING)

    if not args.family:
        logging.error("Error: Specify families with -f.")
        return

    config = {}
    if args.config:
        try:
            with open(args.config, "rb") as f:
                config = toml.load(f)
            logging.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logging.error(f"Failed to parse config file: {e}")
            sys.exit(1)

    hmm_configs = {name: path for name, path in args.family}
    analyzer = GeneFamilyAnalyzer(args.input, args.output, hmm_configs, 
                                  cds_dir=args.cds,
                                  loc_dir=args.loc,
                                  tree=args.tree,
                                  threads=args.threads,
                                  force=args.force,
                                  config=config,
                                  phe_file=args.phe,
                                  search_method=args.method)
    analyzer.process()

if __name__ == "__main__":
    main()
