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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Any
import logging

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
        """Calculate verification metrics (AUC, EER, Precision, Recall, F1)"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # EER calculation
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = fpr[eer_idx] * 100
        
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
        
        return {
            'roc_auc': roc_auc,
            'eer': eer,
            'tpr_at_fpr_1e-4': tpr_at_fpr_1e4,
            'tpr_at_fpr_1e-3': tpr_at_fpr_1e3, 
            'tpr_at_fpr_1e-2': tpr_at_fpr_1e2,
            'precision': precision_score(labels, predictions, zero_division=0) * 100,
            'recall': recall_score(labels, predictions, zero_division=0) * 100,
            'f1_score': f1_score(labels, predictions, zero_division=0) * 100,
        }
    
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


def save_results_to_csv(results_dict, output_path):
    """Save evaluation results to CSV file"""
    try:
        df = pd.DataFrame([results_dict])
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def log_results_to_wandb(results_dict, prefix=""):
    """Log evaluation results to Weights & Biases"""
    if not WANDB_AVAILABLE:
        print("Warning: wandb not available, skipping W&B logging")
        return
        
    try:
        # Filter out non-numeric values for W&B
        wandb_dict = {}
        for key, value in results_dict.items():
            if isinstance(value, (int, float, bool)):
                wandb_dict[f"{prefix}{key}"] = value
            elif isinstance(value, str):
                # Log strings as text
                wandb_dict[f"{prefix}{key}"] = value
                
        wandb.log(wandb_dict)
        print(f"Logged {len(wandb_dict)} metrics to W&B")
    except Exception as e:
        print(f"Error logging to W&B: {e}")