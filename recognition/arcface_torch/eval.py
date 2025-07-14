#!/usr/bin/env python3
"""
Comprehensive ArcFace Evaluation on IJB-C Dataset
A production-ready evaluation framework that combines the best of ArcFace and CVLface approaches.

Features:
- YAML configuration system for easy customization
- A100 optimizations (mixed precision, torch.compile, channels_last memory format)
- Automatic hardware detection and optimization
- Backward compatibility with CLI arguments
- HuggingFace and traditional dataset format support

Usage:
    # With YAML configuration (recommended):
    python eval.py --config a100 --model-path ./models/backbone.pth
    
    # With config overrides:
    python eval.py --config default --model-path ./models/backbone.pth \
                   --override data.batch_size=512 performance.mixed_precision=true
    
    # Legacy CLI usage (backward compatible):
    python eval.py --model-path ./pretrained_models/ms1mv3_arcface_r100_fp16/backbone.pth \
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
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import transform as trans
from tqdm import tqdm

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Import config system
try:
    from config_loader import load_config, EvaluationConfig
    CONFIG_SYSTEM_AVAILABLE = True
except ImportError:
    CONFIG_SYSTEM_AVAILABLE = False
    EvaluationConfig = None

# Import local modules
from backbones import get_model
from eval_utils import (
    ComprehensiveMetrics, PerformanceMonitor, image2template_feature, verification,
    read_template_media_list, read_template_pair_list, save_results_to_csv, log_results_to_wandb,
    load_ijbc_metadata_from_hf_dataset, load_hf_dataset_images, log_progress_to_wandb
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

# A100 specific optimizations
if torch.cuda.is_available():
    # Set memory format to channels last for better A100 performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for A100
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass
    
    # Check if A100 is available
    if torch.cuda.get_device_name().startswith('NVIDIA A100'):
        logger.info("🚀 A100 GPU detected - enabling advanced optimizations")


class IJBCDataset(Dataset):
    """Dataset for pre-aligned IJB-C images (traditional format)"""
    
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


class HFIJBCDataset(Dataset):
    """Dataset for pre-aligned IJB-C images in HuggingFace format with A100 optimizations"""
    
    def __init__(self, hf_dataset, faceness_scores, channels_last=False):
        self.hf_dataset = hf_dataset
        self.faceness_scores = faceness_scores
        self.channels_last = channels_last
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """
        Get preprocessed image from HuggingFace dataset with optimizations
        Returns both original and flipped versions for test-time augmentation
        """
        try:
            # Get image from HuggingFace dataset
            item = self.hf_dataset[idx]
            img = item['image']
            
            # Convert PIL image to numpy array
            if hasattr(img, 'convert'):
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32)
            else:
                img = np.array(img, dtype=np.float32)
            
            # Ensure image is 112x112 (should already be pre-aligned)
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112))
            
            # Create flipped version for TTA
            flipped_img = np.fliplr(img)
            
            # Transpose to CHW format (RGB)
            aligned_img = np.transpose(img, (2, 0, 1))  # 3x112x112, RGB
            flipped_img = np.transpose(flipped_img, (2, 0, 1))
            
            # Stack original and flipped
            input_blob = np.stack([aligned_img, flipped_img], axis=0)  # 2x3x112x112
            
            # Convert to tensor and set memory format if requested
            input_tensor = torch.from_numpy(input_blob)
            if self.channels_last:
                input_tensor = input_tensor.to(memory_format=torch.channels_last)
            
            # Get faceness score
            faceness_score = float(self.faceness_scores[idx])
            
            return {
                'images': input_tensor,
                'faceness_score': faceness_score,
                'img_name': f'image_{idx:06d}'
            }
            
        except Exception as e:
            logger.warning(f"Error processing image {idx}: {e}")
            # Return dummy data in case of error
            dummy_img = torch.zeros((2, 3, 112, 112), dtype=torch.float32)
            if self.channels_last:
                dummy_img = dummy_img.to(memory_format=torch.channels_last)
            return {
                'images': dummy_img,
                'faceness_score': 0.0,
                'img_name': f'error_{idx:06d}'
            }


def load_model(model_path, network, num_features=512, config=None):
    """Load ArcFace model from checkpoint with optional optimizations"""
    logger.info(f"Loading model: {network} from {model_path}")
    
    try:
        # Determine optimization settings
        if config and CONFIG_SYSTEM_AVAILABLE:
            fp16 = config.get('model.fp16', False)
            channels_last = config.get('evaluation.channels_last', False)
            compile_model = config.get('evaluation.compile_model', False)
            compile_mode = config.get('performance.compile_mode', 'default')
        else:
            fp16 = False
            channels_last = False
            compile_model = False
            compile_mode = 'default'
        
        # Load model architecture
        model = get_model(network, dropout=0, fp16=fp16, num_features=num_features)
        
        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Move to device and set eval mode
        model = model.to(device)
        
        # Set memory format for A100 optimization
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
            logger.info("Model converted to channels_last format")
        
        model.eval()
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        
        # Compile model for optimization
        if compile_model:
            logger.info(f"Compiling model with mode: {compile_mode}")
            model = torch.compile(model, mode=compile_mode)
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def extract_features(model, dataloader, use_flip_test=True, use_norm_score=True, use_detector_score=True, config=None, monitor=None, dataset=None, start_time=None):
    """
    Extract features from all images following ArcFace protocol with optional optimizations
    """
    logger.info("Starting feature extraction...")
    
    img_feats = []
    faceness_scores = []
    
    # Mixed precision setup
    if config and CONFIG_SYSTEM_AVAILABLE:
        use_amp = config.get('performance.mixed_precision', False)
        channels_last = config.get('evaluation.channels_last', False)
        log_interval = config.get('monitoring.log_interval', 100)
        clear_cache_interval = config.get('performance.clear_cache_interval', 50)
    else:
        use_amp = False
        channels_last = False
        log_interval = 100
        clear_cache_interval = 50
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            try:
                images = batch['images']  # [batch_size, 2, 3, 112, 112]
                faceness = batch['faceness_score']
                
                batch_size = images.shape[0]
                
                # Reshape to [batch_size*2, 3, 112, 112] for processing both original and flipped
                images = images.view(-1, 3, 112, 112)
                images = images.to(device, non_blocking=True)
                
                # Set memory format if enabled
                if channels_last:
                    images = images.to(memory_format=torch.channels_last)
                
                # Mixed precision forward pass
                if use_amp:
                    with torch.cuda.amp.autocast():
                        # Normalize images: [0,255] -> [-1,1]
                        images = images.div(255.0).sub(0.5).div(0.5)
                        features = model(images)  # [batch_size*2, feature_dim]
                else:
                    # Normalize images: [0,255] -> [-1,1]
                    images = images.div(255.0).sub(0.5).div(0.5)
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
                if batch_idx % log_interval == 0:
                    logger.info(f"Processed {batch_idx * dataloader.batch_size} images")
                    
                    # Log progress to W&B for chart visualization
                    if WANDB_AVAILABLE and wandb.run is not None and start_time is not None:
                        current_images = batch_idx * dataloader.batch_size
                        current_time = time.time() - start_time
                        current_fps = current_images / current_time if current_time > 0 else 0
                        
                        # Get current memory usage if monitor is available
                        memory_stats = {}
                        if monitor is not None:
                            memory_stats = monitor.get_current_memory_usage()
                        
                        progress_metrics = {
                            'images_processed': current_images,
                            'current_fps': current_fps,
                            'elapsed_time_seconds': current_time,
                            'batch_idx': batch_idx,
                        }
                        
                        # Add progress percentage if dataset is available
                        if dataset is not None:
                            progress_metrics['progress_percent'] = (current_images / len(dataset)) * 100
                        
                        # Add memory metrics if available
                        if memory_stats:
                            progress_metrics.update({
                                'current_ram_mb': memory_stats.get('ram_mb', 0),
                                'current_gpu_mb': memory_stats.get('gpu_mb', 0),
                                'current_cpu_percent': memory_stats.get('cpu_percent', 0),
                            })
                        
                        log_progress_to_wandb(batch_idx, progress_metrics)
                # Clear cache periodically for memory efficiency
                if batch_idx % clear_cache_interval == 0:
                    torch.cuda.empty_cache()
                    
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


def setup_wandb(args, config=None):
    """Initialize Weights & Biases logging with config support"""
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available, skipping initialization")
        return False
    
    # Determine W&B settings
    if config and CONFIG_SYSTEM_AVAILABLE:
        wandb_config = config.wandb
        project = wandb_config.get('project') or args.wandb_project
        # Prioritize environment variable over config values
        entity = args.wandb_entity if args.wandb_entity else wandb_config.get('entity')
        tags = wandb_config.get('tags', [])
        
        # Debug logging for entity selection
        logger.info(f"W&B entity selection - args.wandb_entity: '{args.wandb_entity}', config.entity: '{wandb_config.get('entity')}', selected: '{entity}'")
    else:
        project = args.wandb_project
        entity = args.wandb_entity
        tags = []
    
    # Skip W&B if no project specified
    if not project:
        logger.info("No W&B project specified, skipping W&B logging")
        return False
    
    try:
        # Set W&B token from environment if available
        wandb_token = os.getenv('WANDB_TOKEN')
        if wandb_token:
            os.environ['WANDB_API_KEY'] = wandb_token
        
        # Create run name
        if config:
            network = config.get('model.network', args.network)
            target = config.get('data.target', args.target)
            optimization = "A100" if config.get('performance.mixed_precision') else "Standard"
            run_name = f"{network}_{Path(args.model_path).stem}_{target}_{optimization}"
        else:
            run_name = f"{args.network}_{Path(args.model_path).stem}_{args.target}"
        
        # Prepare config data
        run_config = {
            'model_path': args.model_path,
            'network': args.network,
            'target': args.target,
            'batch_size': args.batch_size,
            'use_flip_test': args.use_flip_test,
            'use_norm_score': args.use_norm_score,
            'use_detector_score': args.use_detector_score,
        }
        
        # Add full config if available
        if config:
            run_config.update(config.raw)
        
        # Initialize W&B with environment-based configuration
        init_kwargs = {
            'project': project,
            'name': run_name,
            'config': run_config,
            'tags': tags + [args.network, args.target, "comprehensive", "face-recognition"]
        }
        
        # Add entity if specified
        if entity:
            init_kwargs['entity'] = entity
        
        wandb.init(**init_kwargs)
        logger.info(f"W&B initialized: {run_name}")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        return False


def optimize_gpu_settings(config):
    """Apply GPU optimizations based on configuration"""
    if not torch.cuda.is_available():
        return
        
    # Set memory fraction
    memory_fraction = config.get('performance.cuda_memory_fraction', 0.9)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    # Set matmul precision
    matmul_precision = config.get('performance.matmul_precision', 'high')
    if matmul_precision == 'high':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    logger.info(f"Applied GPU optimizations: memory_fraction={memory_fraction}, matmul_precision={matmul_precision}")


def detect_and_apply_a100_optimizations(config):
    """Detect A100 GPU and apply optimizations automatically"""
    if not torch.cuda.is_available():
        return config
        
    # Check if A100 is available
    if torch.cuda.get_device_name().startswith('NVIDIA A100'):
        logger.info("🚀 A100 GPU detected - applying automatic optimizations")
        
        # Apply A100 optimizations if not already configured
        if not config.get('performance.mixed_precision'):
            config.set('performance.mixed_precision', True)
            logger.info("Enabled mixed precision for A100")
        
        if not config.get('evaluation.channels_last'):
            config.set('evaluation.channels_last', True)
            logger.info("Enabled channels_last memory format for A100")
        
        if not config.get('evaluation.compile_model'):
            config.set('evaluation.compile_model', True)
            logger.info("Enabled model compilation for A100")
        
        # Increase batch size if not set
        if config.get('data.batch_size', 64) < 128:
            config.set('data.batch_size', 256)
            logger.info("Increased batch size to 256 for A100")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Comprehensive ArcFace Evaluation on IJB-C')
    
    # Config arguments (new system)
    parser.add_argument('--config',
                       help='Configuration name (from run_configs/) - enables YAML config system')
    parser.add_argument('--override', action='append', default=[],
                       help='Config overrides in key=value format (e.g., data.batch_size=512)')
    
    # Model arguments
    parser.add_argument('--model-path',
                       default=None,
                       help='Path to model checkpoint (backbone.pth). Can be relative to MODEL_ROOT env var.')
    parser.add_argument('--network', default='r100', help='Model architecture (r18, r34, r50, r100, etc.)')
    parser.add_argument('--num-features', type=int, default=512, help='Feature dimension')
    
    # Data arguments
    parser.add_argument('--image-path',
                       default=os.getenv('IJBC_DATASET_PATH'),
                       help='Path to IJB-C dataset directory (default: IJBC_DATASET_PATH env var)')
    parser.add_argument('--target', default='IJBC', choices=['IJBC', 'IJBB'], help='Target dataset')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--use-flip-test', action='store_true', default=True, help='Use flip test augmentation')
    parser.add_argument('--use-norm-score', action='store_true', default=True, help='Use norm score (TestMode N1)')
    parser.add_argument('--use-detector-score', action='store_true', default=True, help='Use detector score (TestMode D1)')
    
    # Output arguments
    parser.add_argument('--result-dir',
                       default=os.getenv('RESULT_DIR', './eval_results'),
                       help='Directory to save results (default: RESULT_DIR env var or ./eval_results)')
    parser.add_argument('--job', default='arcface_comprehensive', help='Job name for output files')
    
    # Logging arguments
    parser.add_argument('--wandb-project',
                       default=os.getenv('WANDB_PROJECT', 'arcface-ijbc-comprehensive'),
                       help='W&B project name (default: WANDB_PROJECT env var)')
    parser.add_argument('--wandb-entity',
                       default=os.getenv('WANDB_TEAM'),
                       help='W&B entity name (default: WANDB_TEAM env var)')
    
    args = parser.parse_args()
    
    # Load configuration if specified
    config = None
    if args.config and CONFIG_SYSTEM_AVAILABLE:
        # Parse overrides
        overrides = {}
        for override in args.override:
            if '=' not in override:
                continue
            key, value = override.split('=', 1)
            # Try to convert to appropriate type
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
            overrides[key] = value
        
        # Load configuration
        try:
            config = load_config(args.config, **overrides)
            logger.info(f"🔧 Loaded configuration: {args.config}")
            if overrides:
                logger.info(f"Applied overrides: {overrides}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Falling back to CLI arguments")
            config = None
    elif args.config:
        logger.warning("Config system not available (missing config_loader). Using CLI arguments.")
    
    # Auto-detect and apply A100 optimizations if using config system
    if config:
        config = detect_and_apply_a100_optimizations(config)
        optimize_gpu_settings(config)
        
        # Override CLI arguments with config values
        args.network = config.get('model.network', args.network)
        args.num_features = config.get('model.num_features', args.num_features)
        args.target = config.get('data.target', args.target)
        args.batch_size = config.get('data.batch_size', args.batch_size)
        args.num_workers = config.get('data.num_workers', args.num_workers)
        args.use_flip_test = config.get('evaluation.use_flip_test', args.use_flip_test)
        args.use_norm_score = config.get('evaluation.use_norm_score', args.use_norm_score)
        args.use_detector_score = config.get('evaluation.use_detector_score', args.use_detector_score)
        args.result_dir = config.get('output.result_dir', args.result_dir)
        args.job = config.get('output.job', args.job)
        args.wandb_project = config.get('wandb.project', args.wandb_project)
        args.wandb_entity = config.get('wandb.entity', args.wandb_entity)
    
    # Handle model path with MODEL_ROOT support
    if args.model_path is None:
        parser.error("--model-path is required")
    
    # If model path is relative and MODEL_ROOT is set, make it relative to MODEL_ROOT
    if not os.path.isabs(args.model_path):
        model_root = os.getenv('MODEL_ROOT')
        if model_root:
            args.model_path = os.path.join(model_root, args.model_path)
    
    # Validate required paths
    if args.image_path is None:
        parser.error("--image-path is required (or set IJBC_DATASET_PATH environment variable)")
    
    if not os.path.exists(args.image_path):
        parser.error(f"Dataset path does not exist: {args.image_path}")
    
    if not os.path.exists(args.model_path):
        parser.error(f"Model path does not exist: {args.model_path}")
    
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
    if config:
        logger.info(f"Config: {args.config}")
        logger.info(f"Mixed Precision: {config.get('performance.mixed_precision', False)}")
        logger.info(f"Batch Size: {args.batch_size}")
    logger.info("="*80)
    
    # Initialize W&B
    wandb_enabled = setup_wandb(args, config)
    
    # Initialize performance monitoring
    monitor_interval = config.get('monitoring.memory_monitor_interval', 1.0) if config else 1.0
    monitor = PerformanceMonitor(logger=logger)
    monitor.monitor_interval = monitor_interval
    monitor.start()
    
    # Initialize metrics calculator
    metrics_calc = ComprehensiveMetrics()
    
    try:
        # Load model
        model = load_model(args.model_path, args.network, args.num_features, config)
        
        # Load IJB-C metadata - detect format (HuggingFace vs traditional)
        logger.info("Loading IJB-C metadata...")
        metadata_path = os.path.join(args.image_path, 'metadata.pt')
        meta_dir = os.path.join(args.image_path, 'meta')
        
        if os.path.exists(metadata_path):
            # HuggingFace dataset format
            logger.info("Detected HuggingFace dataset format")
            
            # Load metadata from HuggingFace format
            templates, medias, p1, p2, label, faceness_scores = load_ijbc_metadata_from_hf_dataset(args.image_path)
            
            # Load HuggingFace dataset
            hf_dataset = load_hf_dataset_images(args.image_path)
            
            # Create HuggingFace dataset
            channels_last = config.get('evaluation.channels_last', False) if config else False
            dataset = HFIJBCDataset(hf_dataset, faceness_scores, channels_last=channels_last)
            
        elif os.path.exists(meta_dir):
            # Traditional IJB-C format
            logger.info("Detected traditional IJB-C dataset format")
            
            # Load template-media relationships
            templates, medias = read_template_media_list(
                os.path.join(meta_dir, f'{args.target.lower()}_face_tid_mid.txt'))
            
            # Load template pairs for verification
            p1, p2, label = read_template_pair_list(
                os.path.join(meta_dir, f'{args.target.lower()}_template_pair_label.txt'))
            
            # Load image list
            img_list_path = os.path.join(meta_dir, f'{args.target.lower()}_name_5pts_score.txt')
            with open(img_list_path, 'r') as f:
                files_list = f.readlines()
            
            # Create traditional dataset
            dataset = IJBCDataset(
                image_path=os.path.join(args.image_path, 'loose_crop'),
                files_list=files_list
            )
        else:
            raise FileNotFoundError(
                f"No valid IJB-C dataset found at {args.image_path}. "
                f"Expected either metadata.pt (HuggingFace format) or meta/ directory (traditional format)"
            )
        
        logger.info(f"Loaded {len(templates)} images, {len(np.unique(templates))} templates")
        logger.info(f"Loaded {len(p1)} verification pairs")
        
        # Create optimized dataloader
        dataloader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }
        
        # Add optimization parameters if config available
        if config:
            dataloader_kwargs.update({
                'prefetch_factor': config.get('data.prefetch_factor', 2),
                'persistent_workers': config.get('data.persistent_workers', True)
            })
        
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        logger.info(f"Created dataset with {len(dataset)} images")
        logger.info(f"DataLoader: batch_size={dataloader.batch_size}, num_workers={dataloader.num_workers}")
        
        # Extract features
        eval_start_time = time.time()
        img_feats, faceness_scores = extract_features(
            model, dataloader,
            use_flip_test=args.use_flip_test,
            use_norm_score=args.use_norm_score,
            use_detector_score=args.use_detector_score,
            config=config,
            monitor=monitor,
            dataset=dataset,
            start_time=eval_start_time
        )
        
        feature_extraction_time = time.time() - eval_start_time
        
        # Clear model from memory if requested
        if config and config.get('performance.clear_cache_between_models', False):
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
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
            
            # Config info
            'config_name': args.config if args.config else 'CLI',
            'optimization_level': 'A100' if (config and config.get('performance.mixed_precision')) else 'Standard',
            
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
            
            # Optimization settings
            'mixed_precision': config.get('performance.mixed_precision', False) if config else False,
            'channels_last': config.get('evaluation.channels_last', False) if config else False,
            'compiled_model': config.get('evaluation.compile_model', False) if config else False,
            
            # Timestamps
            'evaluation_timestamp': datetime.now().isoformat(),
            'feature_extraction_time_seconds': feature_extraction_time,
        }
        
        # Save results to CSV with timestamp
        result_file = os.path.join(args.result_dir, f'{args.job}_{args.target.lower()}_results_{timestamp}.csv')
        save_results_to_csv(comprehensive_results, result_file)
        
        # Log to W&B
        if wandb_enabled:
            log_results_to_wandb(comprehensive_results)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"🎯 Model: {args.network}")
        logger.info(f"🔧 Config: {args.config if args.config else 'CLI'}")
        logger.info(f"⚡ Optimization: {comprehensive_results['optimization_level']}")
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
        logger.info(f"🔧 Batch Size: {args.batch_size}")
        if config:
            logger.info(f"🎛️  Mixed Precision: {'✅' if config.get('performance.mixed_precision') else '❌'}")
            logger.info(f"🎛️  Channels Last: {'✅' if config.get('evaluation.channels_last') else '❌'}")
            logger.info(f"🎛️  Model Compiled: {'✅' if config.get('evaluation.compile_model') else '❌'}")
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