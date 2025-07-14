"""
Evaluation utilities for ArcFace models on IJB-C dataset.
Ported and adapted from CVLface evaluation framework for comprehensive metrics calculation.
"""

import os
import time
import psutil
import threading
import numpy as np
import pandas as pd
import torch
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Any
import logging
import io
import base64

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available, GPU memory monitoring will be limited")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, no W&B logging")

# Set matplotlib backend for headless operation
plt.switch_backend('Agg')
sns.set_style("whitegrid")


class ComprehensiveMetrics:
    """Calculate comprehensive face recognition metrics"""
    
    @staticmethod
    def calculate_identification_metrics(score_matrix, label_matrix, ranks=[1, 5, 10]):
        """Calculate identification accuracy metrics (Rank-1, Rank-5, Rank-10)"""
        results = {}
        sorted_indices = np.argsort(score_matrix, axis=1)[:, ::-1]
        
        for rank in ranks:
            correct = 0
            total = score_matrix.shape[0]
            
            for i in range(total):
                top_k = sorted_indices[i, :rank]
                if np.any(label_matrix[i, top_k]):
                    correct += 1
                    
            results[f'rank_{rank}_accuracy'] = (correct / total) * 100
            
        return results
    
    @staticmethod
    def calculate_verification_metrics(scores, labels):
        """Calculate verification metrics (AUC, EER, Precision, Recall, F1) with curves"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # EER calculation
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = fpr[eer_idx] * 100
        eer_threshold = thresholds[eer_idx]
        
        # TPR at specific FPR values
        def get_tpr_at_fpr(target_fpr):
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                return tpr[idx[-1]] * 100
            return 0.0
        
        tpr_at_fpr_1e4 = get_tpr_at_fpr(1e-4)
        tpr_at_fpr_1e3 = get_tpr_at_fpr(1e-3)
        tpr_at_fpr_1e2 = get_tpr_at_fpr(1e-2)
        
        # Best F1 threshold
        precision, recall, f1_thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1_idx = np.nanargmax(f1_scores)
        best_threshold = f1_thresholds[best_f1_idx] if len(f1_thresholds) > best_f1_idx else f1_thresholds[-1]
        
        predictions = (scores >= best_threshold).astype(int)
        
        # Store curve data for visualization
        curve_data = {
            'fpr': fpr,
            'tpr': tpr,
            'fnr': fnr,
            'thresholds': thresholds,
            'precision': precision,
            'recall': recall,
            'f1_thresholds': f1_thresholds,
            'f1_scores': f1_scores,
            'eer_threshold': eer_threshold,
            'best_f1_threshold': best_threshold
        }
        
        return {
            'roc_auc': roc_auc,
            'eer': eer,
            'tpr_at_fpr_1e-4': tpr_at_fpr_1e4,
            'tpr_at_fpr_1e-3': tpr_at_fpr_1e3,
            'tpr_at_fpr_1e-2': tpr_at_fpr_1e2,
            'precision': precision_score(labels, predictions, zero_division=0) * 100,
            'recall': recall_score(labels, predictions, zero_division=0) * 100,
            'f1_score': f1_score(labels, predictions, zero_division=0) * 100,
            'curve_data': curve_data,
        }
    
    @staticmethod
    def create_roc_curve_plot(fpr, tpr, roc_auc, eer, model_name="ArcFace"):
        """Create ROC curve visualization"""
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        
        # Mark EER point
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=8, label=f'EER = {eer:.2f}%')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Plot log-scale ROC for better FPR visualization
        plt.subplot(2, 2, 2)
        plt.semilogx(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=8, label=f'EER = {eer:.2f}%')
        plt.xlim([1e-4, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (log scale)')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (Log Scale) - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Plot DET curve (Detection Error Tradeoff)
        plt.subplot(2, 2, 3)
        fnr = 1 - tpr
        plt.loglog(fpr * 100, fnr * 100, color='red', lw=2, label='DET curve')
        plt.plot(fpr[eer_idx] * 100, fnr[eer_idx] * 100, 'ro', markersize=8, label=f'EER = {eer:.2f}%')
        plt.xlabel('False Accept Rate (%)')
        plt.ylabel('False Reject Rate (%)')
        plt.title(f'DET Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot score distribution
        plt.subplot(2, 2, 4)
        # Note: This would need scores for genuine and impostor pairs to be meaningful
        # For now, show a placeholder
        plt.text(0.5, 0.5, 'Score Distribution\n(Requires genuine/impostor scores)',
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title(f'Score Distribution - {model_name}')
        
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def create_performance_summary_plot(metrics, model_name="ArcFace"):
        """Create comprehensive performance summary visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Key metrics bar chart
        ax1 = axes[0, 0]
        key_metrics = ['roc_auc', 'tpr_at_fpr_1e-4', 'tpr_at_fpr_1e-3', 'tpr_at_fpr_1e-2']
        key_values = [metrics.get(k, 0) for k in key_metrics]
        key_labels = ['ROC AUC', 'TPR@FPR=1e-4', 'TPR@FPR=1e-3', 'TPR@FPR=1e-2']
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax1.bar(key_labels, key_values, color=colors, alpha=0.8)
        ax1.set_title(f'Key Verification Metrics - {model_name}')
        ax1.set_ylabel('Score (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, key_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Speed metrics
        ax2 = axes[0, 1]
        speed_metrics = ['fps', 'ms_per_image']
        speed_values = [metrics.get('speed_fps', 0), metrics.get('speed_ms_per_image', 0)]
        speed_labels = ['FPS', 'ms/image']
        
        ax2.bar(speed_labels, speed_values, color=['#4CAF50', '#FF9800'], alpha=0.8)
        ax2.set_title('Speed Performance')
        ax2.set_ylabel('Speed')
        
        # 3. Resource utilization
        ax3 = axes[0, 2]
        resource_labels = ['GPU Memory\n(GB)', 'RAM Peak\n(GB)', 'CPU Avg\n(%)']
        resource_values = [
            metrics.get('resource_gpu_memory_peak_gb', 0),
            metrics.get('resource_ram_peak_mb', 0) / 1024,  # Convert to GB
            metrics.get('resource_cpu_usage_avg_percent', 0)
        ]
        
        colors_resource = ['#E91E63', '#9C27B0', '#3F51B5']
        ax3.bar(resource_labels, resource_values, color=colors_resource, alpha=0.8)
        ax3.set_title('Resource Utilization')
        ax3.set_ylabel('Usage')
        
        # 4. Model information
        ax4 = axes[1, 0]
        model_info = [
            f"Model Size: {metrics.get('model_model_size_mb', 0):.1f} MB",
            f"Batch Size: {metrics.get('batch_size', 'N/A')}",
            f"Mixed Precision: {'✓' if metrics.get('mixed_precision', False) else '✗'}",
            f"Compiled: {'✓' if metrics.get('compiled_model', False) else '✗'}",
            f"Total Images: {metrics.get('total_images', 'N/A'):,}",
            f"Templates: {metrics.get('total_templates', 'N/A'):,}"
        ]
        
        ax4.text(0.05, 0.95, '\n'.join(model_info), transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_title('Model Configuration')
        ax4.axis('off')
        
        # 5. Evaluation timeline
        ax5 = axes[1, 1]
        timeline_data = {
            'Feature Extraction': metrics.get('feature_extraction_time_seconds', 0),
            'Template Aggregation': metrics.get('template_aggregation_time', 0),
            'Metric Calculation': metrics.get('metric_calculation_time', 0)
        }
        
        # Create a simple timeline visualization
        times = list(timeline_data.values())
        labels = list(timeline_data.keys())
        cumulative_times = np.cumsum([0] + times[:-1])
        
        for i, (time, label) in enumerate(zip(times, labels)):
            if time > 0:
                ax5.barh(i, time, left=cumulative_times[i], alpha=0.8)
                ax5.text(cumulative_times[i] + time/2, i, f'{time:.1f}s',
                        ha='center', va='center', fontweight='bold')
        
        ax5.set_yticks(range(len(labels)))
        ax5.set_yticklabels(labels)
        ax5.set_xlabel('Time (seconds)')
        ax5.set_title('Evaluation Timeline')
        
        # 6. Performance comparison (placeholder for multiple models)
        ax6 = axes[1, 2]
        ax6.text(0.5, 0.5, 'Performance\nComparison\n(Multiple Models)',
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Model Comparison')
        ax6.axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def calculate_speed_metrics(total_images, total_time):
        """Calculate speed and throughput metrics"""
        if total_time > 0:
            fps = total_images / total_time
            return {
                'fps': fps,
                'ms_per_image': (total_time * 1000) / total_images,
                'is_realtime': fps >= 30,
                'total_images': total_images,
                'total_time_seconds': total_time,
            }
        return {
            'fps': 0, 
            'ms_per_image': float('inf'), 
            'is_realtime': False,
            'total_images': total_images,
            'total_time_seconds': total_time,
        }
    
    @staticmethod
    def get_model_size_metrics(model_path):
        """Calculate model storage metrics"""
        try:
            if os.path.exists(model_path):
                size_bytes = os.path.getsize(model_path)
                return {
                    'model_size_mb': size_bytes / (1024 ** 2),
                    'model_size_gb': size_bytes / (1024 ** 3),
                }
            else:
                print(f"Warning: Model file not found at {model_path}")
                return {'model_size_mb': 0, 'model_size_gb': 0}
        except Exception as e:
            print(f"Error calculating model size for {model_path}: {e}")
            return {'model_size_mb': 0, 'model_size_gb': 0}


class PerformanceMonitor:
    """Monitor system performance during evaluation with detailed memory tracking"""
    
    def __init__(self, logger=None):
        self.start_time = None
        self.monitoring = False
        self.logger = logger or logging.getLogger(__name__)
        
        # Memory tracking arrays
        self.cpu_usage = []
        self.ram_usage_mb = []
        self.gpu_memory_mb = []
        self.timestamps = []
        
        # Peak values
        self.gpu_memory_peak = 0
        self.ram_peak_mb = 0
        
        # Monitoring interval (seconds)
        self.monitor_interval = 1.0
        
        # Initialize pynvml for accurate GPU memory tracking
        self.gpu_handle = None
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.logger.info("PerformanceMonitor: Using pynvml for accurate GPU memory tracking")
            except Exception as e:
                self.logger.warning(f"Failed to initialize pynvml: {e}")
        
    def start(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.monitoring = True
        
        # Reset tracking arrays
        self.cpu_usage = []
        self.ram_usage_mb = []
        self.gpu_memory_mb = []
        self.timestamps = []
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Start monitoring thread
        threading.Thread(target=self._monitor_resources, daemon=True).start()
        
    def stop(self):
        """Stop performance monitoring and return summary"""
        self.monitoring = False
        duration = time.time() - self.start_time
        
        # Calculate peak GPU memory using pynvml for accurate total GPU memory
        if self.gpu_handle:
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.gpu_memory_peak = memory_info.used / (1024**3)  # Convert to GB
            except Exception as e:
                self.logger.warning(f"Failed to get GPU memory info: {e}")
                self.gpu_memory_peak = 0
            
        # Calculate peak RAM usage
        self.ram_peak_mb = max(self.ram_usage_mb) if self.ram_usage_mb else 0
        
        # Calculate averages
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0
        avg_ram_mb = np.mean(self.ram_usage_mb) if self.ram_usage_mb else 0
        avg_gpu_mb = np.mean(self.gpu_memory_mb) if self.gpu_memory_mb else 0
        
        return {
            'total_duration_seconds': duration,
            'total_duration_minutes': duration / 60,
            'gpu_memory_peak_gb': self.gpu_memory_peak,
            'gpu_memory_peak_mb': self.gpu_memory_peak * 1024,
            'gpu_memory_avg_mb': avg_gpu_mb,
            'ram_peak_mb': self.ram_peak_mb,
            'ram_avg_mb': avg_ram_mb,
            'cpu_usage_avg_percent': avg_cpu,
            'memory_samples_count': len(self.timestamps)
        }
        
    def _monitor_resources(self):
        """Monitor CPU, RAM, and GPU memory usage over time"""
        while self.monitoring:
            try:
                current_time = time.time() - self.start_time
                self.timestamps.append(current_time)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # RAM usage in MB
                ram_info = psutil.virtual_memory()
                ram_used_mb = ram_info.used / (1024**2)
                self.ram_usage_mb.append(ram_used_mb)
                
                # GPU memory usage in MB - use pynvml for accurate total GPU memory
                gpu_memory_mb = 0
                if self.gpu_handle:
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_memory_mb = memory_info.used / (1024**2)
                    except Exception:
                        gpu_memory_mb = 0
                
                self.gpu_memory_mb.append(gpu_memory_mb)
                
                # Log to console periodically (every 30 seconds)
                if len(self.timestamps) % 30 == 0:
                    self.logger.info(
                        f"Memory Monitor - Time: {current_time:.1f}s, "
                        f"RAM: {ram_used_mb:.1f}MB, "
                        f"GPU: {gpu_memory_mb:.1f}MB, "
                        f"CPU: {cpu_percent:.1f}%"
                    )
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.warning(f"Memory monitoring error: {e}")
                time.sleep(self.monitor_interval)
                
    def get_current_memory_usage(self):
        """Get current memory usage snapshot"""
        try:
            ram_info = psutil.virtual_memory()
            ram_used_mb = ram_info.used / (1024**2)
            
            gpu_memory_mb = 0
            if self.gpu_handle:
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_memory_mb = memory_info.used / (1024**2)
                except Exception:
                    gpu_memory_mb = 0
                
            cpu_percent = psutil.cpu_percent()
            
            return {
                'ram_mb': ram_used_mb,
                'gpu_mb': gpu_memory_mb,
                'cpu_percent': cpu_percent,
                'timestamp': time.time() - (self.start_time or time.time())
            }
        except Exception:
            return {'ram_mb': 0, 'gpu_mb': 0, 'cpu_percent': 0, 'timestamp': 0}


def image2template_feature(img_feats=None, templates=None, medias=None):
    """
    Aggregate image features into template features.
    Ported from ArcFace eval_ijbc.py with enhancements.
    
    Args:
        img_feats: [number_image x feats_dim] image features
        templates: template id for each image
        medias: media id for each image
        
    Returns:
        template_norm_feats: normalized template features
        unique_templates: unique template ids
    """
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)]
                
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
            
    # Normalize template features
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    """
    Compute set-to-set similarity scores for verification.
    Ported from ArcFace eval_ijbc.py.
    
    Args:
        template_norm_feats: normalized template features
        unique_templates: unique template ids
        p1, p2: template pairs for verification
        
    Returns:
        score: cosine similarity scores for each pair
    """
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to memory limitation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_template_media_list(path):
    """Read template and media information from IJB-C metadata"""
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    """Read template pairs for verification from IJB-C metadata"""
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def load_ijbc_metadata_from_hf_dataset(dataset_path):
    """
    Load IJB-C metadata from HuggingFace dataset format.
    
    Args:
        dataset_path: Path to the dataset containing metadata.pt
        
    Returns:
        tuple: (templates, medias, p1, p2, label, faceness_scores)
    """
    metadata_path = os.path.join(dataset_path, 'metadata.pt')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        metadata = torch.load(metadata_path, weights_only=False)
        
        # Extract arrays from metadata
        templates = metadata['templates']
        medias = metadata['medias']
        p1 = metadata['p1']
        p2 = metadata['p2']
        label = metadata['label']
        faceness_scores = metadata['faceness_scores']
        
        # Convert to numpy arrays if they're torch tensors
        if isinstance(templates, torch.Tensor):
            templates = templates.numpy()
        if isinstance(medias, torch.Tensor):
            medias = medias.numpy()
        if isinstance(p1, torch.Tensor):
            p1 = p1.numpy()
        if isinstance(p2, torch.Tensor):
            p2 = p2.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if isinstance(faceness_scores, torch.Tensor):
            faceness_scores = faceness_scores.numpy()
            
        print(f"Loaded IJB-C metadata:")
        print(f"  - Images: {len(templates)}")
        print(f"  - Templates: {len(np.unique(templates))}")
        print(f"  - Verification pairs: {len(p1)}")
        
        return templates, medias, p1, p2, label, faceness_scores
        
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata from {metadata_path}: {e}")


def load_hf_dataset_images(dataset_path):
    """
    Load images from HuggingFace dataset format.
    
    Args:
        dataset_path: Path to the HuggingFace dataset
        
    Returns:
        dataset: HuggingFace dataset object
    """
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        print(f"Loaded HuggingFace dataset with {len(dataset)} images")
        return dataset
    except ImportError:
        raise ImportError("datasets library not found. Install with: pip install datasets")
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace dataset from {dataset_path}: {e}")


def save_results_to_csv(results_dict, output_path):
    """Save evaluation results to CSV file"""
    try:
        df = pd.DataFrame([results_dict])
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def plot_to_wandb_image(figure, name="plot"):
    """Convert matplotlib figure to W&B image"""
    try:
        # Save figure to bytes buffer
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                      facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # Create W&B image
        image = wandb.Image(buf, caption=name)
        plt.close(figure)  # Clean up memory
        return image
    except Exception as e:
        print(f"Error converting plot to W&B image: {e}")
        plt.close(figure)
        return None


def log_results_to_wandb(results_dict, prefix=""):
    """Log evaluation results to Weights & Biases with enhanced visualizations"""
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping W&B logging")
        return
        
    try:
        # Filter out non-numeric values for W&B
        wandb_dict = {}
        curve_data = None
        
        for key, value in results_dict.items():
            if key == 'verification_curve_data':
                curve_data = value
                continue
            elif isinstance(value, (int, float, bool)):
                wandb_dict[f"{prefix}{key}"] = value
            elif isinstance(value, str):
                # Log strings as text
                wandb_dict[f"{prefix}{key}"] = value
        
        # Create and log visualizations if curve data is available
        if curve_data:
            try:
                # Create ROC curve plot
                model_name = results_dict.get('network', 'ArcFace')
                roc_auc = results_dict.get('verification_roc_auc', 0)
                eer = results_dict.get('verification_eer', 0)
                
                roc_figure = ComprehensiveMetrics.create_roc_curve_plot(
                    curve_data['fpr'], curve_data['tpr'], roc_auc, eer, model_name
                )
                roc_image = plot_to_wandb_image(roc_figure, "ROC_DET_Curves")
                if roc_image:
                    wandb_dict["visualization/roc_det_curves"] = roc_image
                
                # Create performance summary plot
                summary_figure = ComprehensiveMetrics.create_performance_summary_plot(
                    results_dict, model_name
                )
                summary_image = plot_to_wandb_image(summary_figure, "Performance_Summary")
                if summary_image:
                    wandb_dict["visualization/performance_summary"] = summary_image
                
                # Log individual curve data as tables for interactive plots
                curve_table = wandb.Table(columns=["FPR", "TPR", "Threshold"], data=[
                    [fpr, tpr, thresh] for fpr, tpr, thresh in zip(
                        curve_data['fpr'][::10], curve_data['tpr'][::10], curve_data['thresholds'][::10]
                    )
                ])
                wandb_dict["curves/roc_data"] = curve_table
                
                # Create interactive ROC plot
                wandb_dict["curves/roc_plot"] = wandb.plot.line_series(
                    xs=curve_data['fpr'][::10],
                    ys=[curve_data['tpr'][::10]],
                    keys=["TPR"],
                    title="Interactive ROC Curve",
                    xname="False Positive Rate"
                )
                
            except Exception as e:
                print(f"Error creating visualizations: {e}")
        
        # Log all metrics
        wandb.log(wandb_dict)
        print(f"Logged {len(wandb_dict)} metrics and visualizations to W&B")
        
    except Exception as e:
        print(f"Error logging to W&B: {e}")


def log_progress_to_wandb(step, metrics_dict, prefix="progress"):
    """Log time series data to W&B for chart visualization"""
    if not WANDB_AVAILABLE:
        return
        
    try:
        # Prepare metrics with step information
        wandb_dict = {"step": step}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float, bool)):
                wandb_dict[f"{prefix}/{key}"] = value
                
        wandb.log(wandb_dict, step=step)
    except Exception as e:
        print(f"Error logging progress to W&B: {e}")


def create_evaluation_report(results_dict, output_dir="./eval_results"):
    """Create a comprehensive evaluation report with visualizations"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract curve data if available
        curve_data = results_dict.get('verification_curve_data')
        model_name = results_dict.get('network', 'ArcFace')
        
        if curve_data:
            # Create and save ROC curves
            roc_auc = results_dict.get('verification_roc_auc', 0)
            eer = results_dict.get('verification_eer', 0)
            
            roc_figure = ComprehensiveMetrics.create_roc_curve_plot(
                curve_data['fpr'], curve_data['tpr'], roc_auc, eer, model_name
            )
            roc_path = os.path.join(output_dir, f'roc_det_curves_{timestamp}.png')
            roc_figure.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close(roc_figure)
            print(f"ROC/DET curves saved to: {roc_path}")
            
            # Create and save performance summary
            summary_figure = ComprehensiveMetrics.create_performance_summary_plot(
                results_dict, model_name
            )
            summary_path = os.path.join(output_dir, f'performance_summary_{timestamp}.png')
            summary_figure.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close(summary_figure)
            print(f"Performance summary saved to: {summary_path}")
        
        # Create markdown report
        report_path = os.path.join(output_dir, f'evaluation_report_{timestamp}.md')
        with open(report_path, 'w') as f:
            f.write(f"# ArcFace Evaluation Report\n\n")
            f.write(f"**Model:** {model_name}\n")
            f.write(f"**Timestamp:** {results_dict.get('evaluation_timestamp', 'N/A')}\n")
            f.write(f"**Dataset:** {results_dict.get('target', 'IJB-C')}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write(f"- **ROC AUC:** {results_dict.get('verification_roc_auc', 0):.4f}\n")
            f.write(f"- **EER:** {results_dict.get('verification_eer', 0):.2f}%\n")
            f.write(f"- **TPR@FPR=1e-4:** {results_dict.get('verification_tpr_at_fpr_1e-4', 0):.2f}%\n")
            f.write(f"- **TPR@FPR=1e-3:** {results_dict.get('verification_tpr_at_fpr_1e-3', 0):.2f}%\n")
            f.write(f"- **TPR@FPR=1e-2:** {results_dict.get('verification_tpr_at_fpr_1e-2', 0):.2f}%\n\n")
            
            f.write("## Speed & Resource Metrics\n\n")
            f.write(f"- **FPS:** {results_dict.get('speed_fps', 0):.1f}\n")
            f.write(f"- **ms/image:** {results_dict.get('speed_ms_per_image', 0):.2f}\n")
            f.write(f"- **GPU Memory Peak:** {results_dict.get('resource_gpu_memory_peak_gb', 0):.2f} GB\n")
            f.write(f"- **RAM Peak:** {results_dict.get('resource_ram_peak_mb', 0):.1f} MB\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Batch Size:** {results_dict.get('batch_size', 'N/A')}\n")
            f.write(f"- **Mixed Precision:** {'✓' if results_dict.get('mixed_precision', False) else '✗'}\n")
            f.write(f"- **Model Compiled:** {'✓' if results_dict.get('compiled_model', False) else '✗'}\n")
            f.write(f"- **Model Size:** {results_dict.get('model_model_size_mb', 0):.1f} MB\n")
            
        print(f"Markdown report saved to: {report_path}")
        
        return {
            'report_path': report_path,
            'roc_path': roc_path if curve_data else None,
            'summary_path': summary_path if curve_data else None
        }
        
    except Exception as e:
        print(f"Error creating evaluation report: {e}")
        return None