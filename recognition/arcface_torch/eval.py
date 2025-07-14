#!/usr/bin/env python3
"""
Comprehensive ArcFace Evaluation on IJB-C Dataset
A production-ready evaluation framework that combines the best of ArcFace and CVLface approaches.

Usage:
    python eval_arcface_full.py --model-path ./pretrained_models/ms1mv3_arcface_r100_fp16/backbone.pth \
                                --network r100 \
                                --image-path /path/to/IJBC_gt_aligned \
                                --result-dir ./eval_results \
                                --wandb-project arcface-ijbc-eval
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import transform as trans
from tqdm import tqdm

# Import local modules
from backbones import get_model
from eval_utils import (
    ComprehensiveMetrics, PerformanceMonitor, image2template_feature, verification,
    read_template_media_list, read_template_pair_list, save_results_to_csv, log_results_to_wandb
)

# Check for optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, no W&B logging")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class IJBCDataset(Dataset):
    """Dataset for pre-aligned IJB-C images"""
    
    def __init__(self, image_path, files_list, transform=None):
        self.image_path = image_path
        self.files_list = files_list
        self.transform = transform
        
        # Standard 5-point landmarks for face alignment (from eval_ijbc.py)
        self.src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        self.src[:, 0] += 8.0
        
    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, idx):
        """
        Get preprocessed image following ArcFace eval_ijbc.py protocol
        Returns both original and flipped versions for test-time augmentation
        """
        line = self.files_list[idx].strip().split(' ')
        img_name = os.path.join(self.image_path, line[0])
        
        try:
            # Load image
            img = cv2.imread(img_name)
            if img is None:
                raise ValueError(f"Could not load image: {img_name}")
            
            # Extract landmarks
            lmk = np.array([float(x) for x in line[1:-1]], dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            
            # Face alignment using similarity transform
            tform = trans.SimilarityTransform()
            tform.estimate(lmk, self.src)
            M = tform.params[0:2, :]
            
            # Warp image to 112x112
            aligned_img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
            
            # Convert BGR to RGB
            aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
            
            # Create flipped version for TTA
            flipped_img = np.fliplr(aligned_img)
            
            # Transpose to CHW format
            aligned_img = np.transpose(aligned_img, (2, 0, 1))  # 3x112x112, RGB
            flipped_img = np.transpose(flipped_img, (2, 0, 1))
            
            # Stack original and flipped
            input_blob = np.stack([aligned_img, flipped_img], axis=0)  # 2x3x112x112
            
            # Extract faceness score
            faceness_score = float(line[-1])
            
            return {
                'images': input_blob.astype(np.float32),
                'faceness_score': faceness_score,
                'img_name': line[0]
            }
            
        except Exception as e:
            logger.warning(f"Error processing {img_name}: {e}")
            # Return dummy data in case of error
            dummy_img = np.zeros((2, 3, 112, 112), dtype=np.float32)
            return {
                'images': dummy_img,
                'faceness_score': 0.0,
                'img_name': line[0] if len(line) > 0 else 'error'
            }


def load_model(model_path, network, num_features=512):
    """Load ArcFace model from checkpoint"""
    logger.info(f"Loading model: {network} from {model_path}")
    
    try:
        # Load model architecture
        model = get_model(network, dropout=0, fp16=False, num_features=num_features)
        
        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Move to device and set eval mode
        model = model.to(device)
        model.eval()
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def extract_features(model, dataloader, use_flip_test=True, use_norm_score=True, use_detector_score=True):
    """
    Extract features from all images following ArcFace protocol
    """
    logger.info("Starting feature extraction...")
    
    img_feats = []
    faceness_scores = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            try:
                images = batch['images']  # [batch_size, 2, 3, 112, 112]
                faceness = batch['faceness_score']
                
                batch_size = images.shape[0]
                
                # Reshape to [batch_size*2, 3, 112, 112] for processing both original and flipped
                images = images.view(-1, 3, 112, 112)
                images = images.to(device)
                
                # Normalize images: [0,255] -> [-1,1]
                images = images.div(255.0).sub(0.5).div(0.5)
                
                # Forward pass
                features = model(images)  # [batch_size*2, feature_dim]
                
                # Reshape back to [batch_size, 2, feature_dim]
                features = features.view(batch_size, 2, -1)
                
                if use_flip_test:
                    # Add original and flipped features (F2 mode from eval_ijbc.py)
                    combined_features = features[:, 0, :] + features[:, 1, :]
                else:
                    # Use only original features
                    combined_features = features[:, 0, :]
                
                # Feature normalization
                if not use_norm_score:
                    # Normalize to unit length (remove norm information)
                    combined_features = torch.nn.functional.normalize(combined_features, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                combined_features = combined_features.cpu().numpy()
                
                img_feats.append(combined_features)
                faceness_scores.extend(faceness.numpy())
                
                # Log progress periodically
                if batch_idx % 100 == 0:
                    logger.info(f"Processed {batch_idx * dataloader.batch_size} images")
                    
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Concatenate all features
    img_feats = np.concatenate(img_feats, axis=0)
    faceness_scores = np.array(faceness_scores)
    
    # Apply detector scores (faceness weighting)
    if use_detector_score:
        logger.info("Applying faceness score weighting")
        img_feats = img_feats * faceness_scores[:, np.newaxis]
    
    logger.info(f"Feature extraction completed. Shape: {img_feats.shape}")
    return img_feats, faceness_scores


def setup_wandb(args):
    """Initialize Weights & Biases logging"""
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available, skipping initialization")
        return False
    
    try:
        run_name = f"{args.network}_{Path(args.model_path).stem}_{args.target}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'model_path': args.model_path,
                'network': args.network,
                'target': args.target,
                'batch_size': args.batch_size,
                'use_flip_test': args.use_flip_test,
                'use_norm_score': args.use_norm_score,
                'use_detector_score': args.use_detector_score,
            },
            tags=[args.network, args.target, "comprehensive", "face-recognition"]
        )
        logger.info(f"W&B initialized: {run_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Comprehensive ArcFace Evaluation on IJB-C')
    
    # Model arguments
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint (backbone.pth)')
    parser.add_argument('--network', default='r100', help='Model architecture (r18, r34, r50, r100, etc.)')
    parser.add_argument('--num-features', type=int, default=512, help='Feature dimension')
    
    # Data arguments
    parser.add_argument('--image-path', required=True, help='Path to IJB-C dataset directory')
    parser.add_argument('--target', default='IJBC', choices=['IJBC', 'IJBB'], help='Target dataset')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--use-flip-test', action='store_true', default=True, help='Use flip test augmentation')
    parser.add_argument('--use-norm-score', action='store_true', default=True, help='Use norm score (TestMode N1)')
    parser.add_argument('--use-detector-score', action='store_true', default=True, help='Use detector score (TestMode D1)')
    
    # Output arguments
    parser.add_argument('--result-dir', default='./eval_results', help='Directory to save results')
    parser.add_argument('--job', default='arcface_comprehensive', help='Job name for output files')
    
    # Logging arguments
    parser.add_argument('--wandb-project', default='arcface-ijbc-comprehensive', help='W&B project name')
    parser.add_argument('--wandb-entity', default=None, help='W&B entity name')
    
    args = parser.parse_args()
    
    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.result_dir, f'eval_log_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*80)
    logger.info("🚀 Starting Comprehensive ArcFace Evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Network: {args.network}")
    logger.info(f"Dataset: {args.target}")
    logger.info(f"Device: {device}")
    logger.info("="*80)
    
    # Initialize W&B
    wandb_enabled = setup_wandb(args)
    
    # Initialize performance monitoring
    monitor = PerformanceMonitor(logger=logger)
    monitor.start()
    
    # Initialize metrics calculator
    metrics_calc = ComprehensiveMetrics()
    
    try:
        # Load model
        model = load_model(args.model_path, args.network, args.num_features)
        
        # Load IJB-C metadata
        logger.info("Loading IJB-C metadata...")
        meta_dir = os.path.join(args.image_path, 'meta')
        
        # Load template-media relationships
        templates, medias = read_template_media_list(
            os.path.join(meta_dir, f'{args.target.lower()}_face_tid_mid.txt'))
        
        # Load template pairs for verification
        p1, p2, label = read_template_pair_list(
            os.path.join(meta_dir, f'{args.target.lower()}_template_pair_label.txt'))
        
        logger.info(f"Loaded {len(templates)} images, {len(np.unique(templates))} templates")
        logger.info(f"Loaded {len(p1)} verification pairs")
        
        # Load image list
        img_list_path = os.path.join(meta_dir, f'{args.target.lower()}_name_5pts_score.txt')
        with open(img_list_path, 'r') as f:
            files_list = f.readlines()
        
        # Create dataset and dataloader
        dataset = IJBCDataset(
            image_path=os.path.join(args.image_path, 'loose_crop'),
            files_list=files_list
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created dataset with {len(dataset)} images")
        
        # Extract features
        eval_start_time = time.time()
        img_feats, faceness_scores = extract_features(
            model, dataloader, 
            use_flip_test=args.use_flip_test,
            use_norm_score=args.use_norm_score,
            use_detector_score=args.use_detector_score
        )
        
        feature_extraction_time = time.time() - eval_start_time
        
        # Aggregate features into templates
        logger.info("Aggregating template features...")
        template_norm_feats, unique_templates = image2template_feature(
            img_feats, templates, medias)
        
        # Calculate verification scores
        logger.info("Computing verification scores...")
        verification_scores = verification(template_norm_feats, unique_templates, p1, p2)
        
        # Calculate comprehensive metrics
        logger.info("Calculating comprehensive metrics...")
        
        # Verification metrics
        verification_metrics = metrics_calc.calculate_verification_metrics(verification_scores, label)
        
        # Speed metrics
        speed_metrics = metrics_calc.calculate_speed_metrics(len(dataset), feature_extraction_time)
        
        # Model size metrics
        model_size_metrics = metrics_calc.get_model_size_metrics(args.model_path)
        
        # Stop performance monitoring
        resource_metrics = monitor.stop()
        
        # Combine all results
        comprehensive_results = {
            # Model info
            'model_path': args.model_path,
            'network': args.network,
            'num_features': args.num_features,
            'target': args.target,
            
            # Verification metrics
            **{f'verification_{k}': v for k, v in verification_metrics.items()},
            
            # Speed metrics
            **{f'speed_{k}': v for k, v in speed_metrics.items()},
            
            # Model size metrics
            **{f'model_{k}': v for k, v in model_size_metrics.items()},
            
            # Resource metrics
            **{f'resource_{k}': v for k, v in resource_metrics.items()},
            
            # Dataset info
            'total_images': len(dataset),
            'total_templates': len(unique_templates),
            'verification_pairs': len(p1),
            
            # Evaluation settings
            'use_flip_test': args.use_flip_test,
            'use_norm_score': args.use_norm_score,
            'use_detector_score': args.use_detector_score,
            'batch_size': args.batch_size,
            
            # Timestamps
            'evaluation_timestamp': datetime.now().isoformat(),
            'feature_extraction_time_seconds': feature_extraction_time,
        }
        
        # Save results to CSV
        result_file = os.path.join(args.result_dir, f'{args.job}_{args.target.lower()}_results.csv')
        save_results_to_csv(comprehensive_results, result_file)
        
        # Log to W&B
        if wandb_enabled:
            log_results_to_wandb(comprehensive_results)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"🎯 Model: {args.network}")
        logger.info(f"📊 {args.target} TPR@FPR=1e-4: {verification_metrics['tpr_at_fpr_1e-4']:.2f}%")
        logger.info(f"📈 ROC AUC: {verification_metrics['roc_auc']:.4f}")
        logger.info(f"⚖️  EER: {verification_metrics['eer']:.2f}%")
        logger.info(f"🎯 F1 Score: {verification_metrics['f1_score']:.2f}%")
        logger.info(f"⚡ FPS: {speed_metrics['fps']:.1f}")
        logger.info(f"🚀 Real-time: {'✅' if speed_metrics['is_realtime'] else '❌'}")
        logger.info(f"💾 GPU Memory Peak: {resource_metrics['gpu_memory_peak_gb']:.2f} GB")
        logger.info(f"🧠 RAM Peak: {resource_metrics['ram_peak_mb']:.1f} MB")
        logger.info(f"⏱️  Total Time: {resource_metrics['total_duration_minutes']:.1f} min")
        logger.info(f"📦 Model Size: {model_size_metrics['model_size_mb']:.1f} MB")
        logger.info("="*80)
        
        logger.info(f"✅ Evaluation completed successfully!")
        logger.info(f"📁 Results saved to: {result_file}")
        logger.info(f"📋 Log saved to: {log_file}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise
    finally:
        if wandb_enabled:
            wandb.finish()


if __name__ == '__main__':
    main()