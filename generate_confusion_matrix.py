"""
Generate Confusion Matrix for DiaNew Diagnostic Results

This script reads the test results CSV and generates confusion matrices
based on the classification logic from the TIM manuscript.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def map_scenario_to_ground_truth(scenario_name: str) -> str:
    """
    Map scenario name to ground truth classification.
    
    Args:
        scenario_name: Name from the 'scenario' column
        
    Returns:
        Ground truth classification label
    """
    mapping = {
        'multipath_thermal': 'Multipath',
        'multipath_smm_thermal': 'SMM',
        'multipath_optimistic': 'Optimism',
        'multipath_permissive': 'Pessimism',
        'multipath_smm_optimistic': 'SMM',
        'multipath_smm_permissive': 'SMM'
    }
    return mapping.get(scenario_name, 'Unknown')

def classify_diagnosis(p_elt: float, nci: float, srd_nll: float, srd_es: float, 
                       nees_ks_pvalue: float, tau_nci: float = 3, 
                       alpha_sig: float = 0.001, nees_chi2_alpha: float = 0.01) -> str:
    """
    Classify diagnosis based on DiaNew metrics following design doc.md logic.
    
    Two-Line Defense Strategy:
    1. First Line: Raw ELT (Bias Detection)
    2. Second Line: NCI-based NMM Detection (if no bias)
    
    Args:
        p_elt: P-value from Energy Location Test
        nci: Normalized Covariance Index (dB)
        srd_nll: Slope Relative Difference for NLL (not used in simplified logic)
        srd_es: Slope Relative Difference for ES (not used in simplified logic)
        nees_ks_pvalue: P-value from KS test on NEES (not used in simplified logic)
        tau_nci: NCI threshold (default 0.5 dB)
        alpha_sig: Significance level for ELT (default 0.05)
        nees_chi2_alpha: Alpha for NEES chi2 test (default 0.01, not used)
        
    Returns:
        Predicted classification label
    """
    # First Line Defense: Raw ELT (Bias Detection)
    # According to design doc.md: "如果 FAIL (p < alpha): 
    # 确认为 System Bias Detected。这是真正的物理故障，无需后续 NMM 诊断，直接报警。"
    if p_elt < alpha_sig:
        return 'SMM'  # System Bias Detected - no need for NMM diagnosis
    
    # Second Line Defense: NMM Detection (only if no bias)
    # According to design doc.md Section 4.2, lines 132-135:
    # "判据：若 NCI > 0.1 (dB)，判定为 Optimistic。
    #        若 NCI < -0.1 (dB)，判定为 Pessimistic。
    #        否则，判定为 Calibrated。"
    if nci > tau_nci:
        return 'Optimism'
    elif nci < -tau_nci:
        return 'Pessimism'
    else:
        return 'Multipath'

def compute_confusion_matrix(df: pd.DataFrame):
    """
    Compute confusion matrix from DataFrame.
    
    Args:
        df: DataFrame with 'ground_truth' and predicted labels
        
    Returns:
        cm: Confusion matrix (numpy array)
        all_labels: List of all unique labels
    """
    # Get all unique labels (sorted for consistent ordering)
    all_labels = sorted(set(df['ground_truth'].unique()) | set(df['predicted'].unique()))
    
    # Generate confusion matrix manually
    n_labels = len(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for _, row in df.iterrows():
        true_idx = label_to_idx[row['ground_truth']]
        pred_idx = label_to_idx[row['predicted']]
        cm[true_idx, pred_idx] += 1
    
    return cm, all_labels

def generate_confusion_matrix(csv_path: str, output_dir: str = 'plots', use_block: bool = False):
    """
    Generate confusion matrix from test results CSV.
    
    Args:
        csv_path: Path to the CSV file with test results
        output_dir: Directory to save output plots
        use_block: If True, use block-averaged metrics; otherwise use raw series metrics
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Map ground truth from scenario names
    df['ground_truth'] = df['scenario'].apply(map_scenario_to_ground_truth)
    
    # Classify predictions based on metrics
    if use_block:
        df['predicted'] = df.apply(
            lambda row: classify_diagnosis(
                p_elt=row['p_elt'],  # Use p_elt instead of elt
                nci=row['nci_block'],
                srd_nll=row['srd_nll_block'],
                srd_es=row['srd_es_block'],
                nees_ks_pvalue=row['nees_ks_pvalue_block']
            ),
            axis=1
        )
        metric_type = 'block'
    else:
        df['predicted'] = df.apply(
            lambda row: classify_diagnosis(
                p_elt=row['p_elt'],  # Use p_elt instead of elt
                nci=row['nci'],
                srd_nll=row['srd_nll'],
                srd_es=row['srd_es'],
                nees_ks_pvalue=row['nees_ks_pvalue']
            ),
            axis=1
        )
        metric_type = 'raw'
    
    # Compute confusion matrix
    cm, all_labels = compute_confusion_matrix(df)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    
    # Print classification report (manually calculated)
    print(f"\n{'='*80}")
    print(f"Classification Report ({metric_type.capitalize()} Metrics)")
    print(f"{'='*80}")
    print(f"{'Label':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 80)
    
    total_correct = 0
    total_samples = len(df)
    
    for i, label in enumerate(all_labels):
        tp = cm[i, i]  # True positives
        fp = np.sum(cm[:, i]) - tp  # False positives (predicted as this but actually other)
        fn = np.sum(cm[i, :]) - tp  # False negatives (actually this but predicted as other)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn
        
        total_correct += tp
        
        print(f"{label:<20} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<12}")
    
    accuracy = total_correct / total_samples
    print("-" * 80)
    print(f"{'Accuracy':<20} {accuracy:<12.3f} {'':<12} {'':<12} {total_samples:<12}")
    
    # Print confusion matrix as table
    print(f"\n{'='*80}")
    print(f"Confusion Matrix ({metric_type.capitalize()} Metrics)")
    print(f"{'='*80}")
    print(cm_df)
    
    # Calculate and print accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    return df, cm, cm_df, all_labels

if __name__ == '__main__':
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate confusion matrix plots from DiaNew CSV results')
    parser.add_argument(
        'csv_path',
        nargs='?',
        default='test_results/dianew_multipath_test_20260118_214954_d4.0h_scale_4.0.csv',
        help='Path to the CSV file (default: %(default)s)'
    )
    parser.add_argument(
        '--show-block',
        action='store_true',
        default=False,
        help='Also compute and plot block-metrics confusion matrix'
    )
    args = parser.parse_args()
    csv_path = args.csv_path
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        raise SystemExit(1)
    
    # Generate confusion matrices for raw metrics (always)
    print("Generating confusion matrix for RAW metrics...")
    df_raw, cm_raw, cm_df_raw, labels_raw = generate_confusion_matrix(csv_path, use_block=False)

    if args.show_block:
        print("\n" + "="*80 + "\n")
        print("Generating confusion matrix for BLOCK metrics...")
        df_block, cm_block, cm_df_block, labels_block = generate_confusion_matrix(csv_path, use_block=True)

        # Get unified labels for both confusion matrices
        all_labels = sorted(set(labels_raw) | set(labels_block))
        n_labels = len(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

        # Expand cm_raw to include all labels
        cm_raw_expanded = np.zeros((n_labels, n_labels), dtype=int)
        for i, label_i in enumerate(labels_raw):
            for j, label_j in enumerate(labels_raw):
                cm_raw_expanded[label_to_idx[label_i], label_to_idx[label_j]] = cm_raw[i, j]

        # Expand cm_block to include all labels
        cm_block_expanded = np.zeros((n_labels, n_labels), dtype=int)
        for i, label_i in enumerate(labels_block):
            for j, label_j in enumerate(labels_block):
                cm_block_expanded[label_to_idx[label_i], label_to_idx[label_j]] = cm_block[i, j]

        # Create figure with two subplots (raw and block)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes_list = [axes[0], axes[1]]
        cms = [cm_raw_expanded, cm_block_expanded]
        titles = ['Confusion Matrix (Raw Metrics)', 'Confusion Matrix (Block Metrics)']
    else:
        # Raw only
        all_labels = labels_raw
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        axes_list = [ax]
        cms = [cm_raw]
        titles = ['Confusion Matrix (Raw Metrics)']

    # Plot matrices
    for ax, cm_plot, title in zip(axes_list, cms, titles):
        im = ax.imshow(cm_plot, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_yticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_yticklabels(all_labels)
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')

        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                ax.text(
                    j, i, int(cm_plot[i, j]),
                    ha="center", va="center", color="black", fontweight='bold'
                )
        plt.colorbar(im, ax=ax, label='Count')

    plt.tight_layout()

    # Save figure
    os.makedirs('plots', exist_ok=True)
    output_path = os.path.join('plots', 'confusion_matrix_combined.png' if args.show_block else 'confusion_matrix_raw.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix figure saved to: {output_path}")

    plt.show()
