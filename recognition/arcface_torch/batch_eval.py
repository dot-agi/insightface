#!/usr/bin/env python3
"""
Batch Evaluation Script for Multiple ArcFace Models on IJB-C
Automates comprehensive evaluation across multiple models with parallel processing support.

Usage:
    python batch_eval_arcface.py --config batch_config.json --max-workers 2
    python batch_eval_arcface.py --create-sample-config  # Generate sample config
"""

import os
import sys
import json
import argparse
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BatchEvaluator:
    """Batch evaluator for multiple ArcFace models"""
    
    def __init__(self, config_path: str):
        """Initialize batch evaluator with configuration"""
        self.config = self.load_config(config_path)
        self.results = []
        self.start_time = time.time()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load evaluation configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['models', 'image_path', 'result_dir']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field in config: {field}")
            
            # Set defaults
            config.setdefault('target', 'IJBC')
            config.setdefault('batch_size', 64)
            config.setdefault('num_workers', 4)
            config.setdefault('use_flip_test', True)
            config.setdefault('use_norm_score', True)
            config.setdefault('use_detector_score', True)
            config.setdefault('wandb_project', 'arcface-batch-eval')
            config.setdefault('wandb_entity', None)
            
            logger.info(f"Loaded configuration for {len(config['models'])} models")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
    
    def run_single_evaluation(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation for a single model"""
        model_name = model_config.get('name', 'unknown')
        model_path = model_config['model_path']
        network = model_config['network']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Evaluating: {model_name}")
        logger.info(f"📁 Model: {model_path}")
        logger.info(f"🏗️  Network: {network}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Validate model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Create model-specific output directory
            model_result_dir = os.path.join(
                self.config['result_dir'],
                f"{model_name}_{network}_{int(time.time())}"
            )
            os.makedirs(model_result_dir, exist_ok=True)
            
            # Build evaluation command
            cmd = [
                sys.executable, "eval.py",
                "--model-path", model_path,
                "--network", network,
                "--image-path", self.config['image_path'],
                "--target", self.config['target'],
                "--result-dir", model_result_dir,
                "--batch-size", str(self.config['batch_size']),
                "--num-workers", str(self.config['num_workers']),
                "--job", f"{model_name}_{network}",
                "--wandb-project", self.config['wandb_project']
            ]
            
            # Add optional arguments
            if self.config['use_flip_test']:
                cmd.append("--use-flip-test")
            if self.config['use_norm_score']:
                cmd.append("--use-norm-score")
            if self.config['use_detector_score']:
                cmd.append("--use-detector-score")
            
            if self.config['wandb_entity']:
                cmd.extend(["--wandb-entity", self.config['wandb_entity']])
            
            # Add model-specific parameters
            if 'num_features' in model_config:
                cmd.extend(["--num-features", str(model_config['num_features'])])
            
            logger.info(f"🔧 Command: {' '.join(cmd)}")
            
            # Run evaluation with timeout
            timeout = model_config.get('timeout', 3600)  # 1 hour default
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                # Load results from CSV
                result_files = list(Path(model_result_dir).glob("*_results.csv"))
                if result_files:
                    results_df = pd.read_csv(result_files[0])
                    model_results = results_df.iloc[0].to_dict()
                    
                    # Add metadata
                    model_results.update({
                        'model_name': model_name,
                        'model_path': model_path,
                        'network': network,
                        'status': 'success',
                        'elapsed_time_seconds': elapsed_time,
                        'result_dir': model_result_dir,
                        'evaluation_timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"✅ {model_name} completed successfully in {elapsed_time:.1f}s")
                    return model_results
                else:
                    logger.error(f"❌ {model_name} failed: No result files found")
                    return self._create_error_result(model_name, model_path, network, 
                                                   "No result files found", elapsed_time)
            else:
                logger.error(f"❌ {model_name} failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return self._create_error_result(model_name, model_path, network, 
                                               result.stderr, elapsed_time)
                
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            logger.error(f"⏰ {model_name} timed out after {elapsed_time:.1f}s")
            return self._create_error_result(model_name, model_path, network, 
                                           "Timeout", elapsed_time)
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ {model_name} failed with exception: {str(e)}")
            return self._create_error_result(model_name, model_path, network, 
                                           str(e), elapsed_time)
    
    def _create_error_result(self, model_name: str, model_path: str, network: str, 
                           error: str, elapsed_time: float) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'model_name': model_name,
            'model_path': model_path,
            'network': network,
            'status': 'failed',
            'error': error,
            'elapsed_time_seconds': elapsed_time,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def run_batch_evaluation(self, max_workers: int = 1) -> List[Dict[str, Any]]:
        """Run evaluation for all models with optional parallel processing"""
        
        models = self.config['models']
        total_models = len(models)
        
        logger.info(f"\n🚀 Starting batch evaluation for {total_models} models")
        logger.info(f"🔧 Max workers: {max_workers}")
        logger.info(f"📁 Result directory: {self.config['result_dir']}")
        
        # Initialize W&B for batch tracking if available
        wandb_run = None
        if WANDB_AVAILABLE:
            try:
                wandb_run = wandb.init(
                    project=f"{self.config['wandb_project']}-batch",
                    name=f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.config,
                    tags=["batch", "comprehensive", self.config['target']]
                )
                logger.info("📊 W&B batch tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
        
        # Run evaluations
        if max_workers == 1:
            # Sequential execution
            logger.info("🔄 Running sequential evaluation")
            for i, model_config in enumerate(models, 1):
                logger.info(f"\n📊 Progress: {i}/{total_models}")
                result = self.run_single_evaluation(model_config)
                self.results.append(result)
        else:
            # Parallel execution
            logger.info(f"⚡ Running parallel evaluation with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(self.run_single_evaluation, model_config): model_config
                    for model_config in models
                }
                
                completed = 0
                for future in as_completed(future_to_model):
                    completed += 1
                    model_config = future_to_model[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                        logger.info(f"📊 Progress: {completed}/{total_models} completed")
                    except Exception as e:
                        model_name = model_config.get('name', 'unknown')
                        logger.error(f"❌ Unexpected error for {model_name}: {e}")
                        error_result = self._create_error_result(
                            model_name, model_config['model_path'], 
                            model_config['network'], str(e), 0
                        )
                        self.results.append(error_result)
        
        # Save and analyze results
        self._save_batch_results()
        self._analyze_results(wandb_run)
        
        if wandb_run:
            wandb_run.finish()
        
        return self.results
    
    def _save_batch_results(self):
        """Save batch results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        json_file = os.path.join(self.config['result_dir'], f'batch_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save successful results as CSV
        successful_results = [r for r in self.results if r['status'] == 'success']
        if successful_results:
            csv_file = os.path.join(self.config['result_dir'], f'batch_results_{timestamp}.csv')
            df = pd.DataFrame(successful_results)
            df.to_csv(csv_file, index=False)
            
            # Create summary table
            summary_cols = [
                'model_name', 'network', 'verification_roc_auc', 'verification_tpr_at_fpr_1e-4',
                'speed_fps', 'model_model_size_mb', 'resource_gpu_memory_peak_gb',
                'elapsed_time_seconds'
            ]
            available_cols = [col for col in summary_cols if col in df.columns]
            
            if available_cols:
                summary_df = df[available_cols].copy()
                if 'verification_roc_auc' in summary_df.columns:
                    summary_df = summary_df.sort_values('verification_roc_auc', ascending=False)
                
                summary_file = os.path.join(self.config['result_dir'], f'summary_{timestamp}.csv')
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"📊 Summary saved to: {summary_file}")
        
        logger.info(f"💾 Batch results saved to: {json_file}")
    
    def _analyze_results(self, wandb_run=None):
        """Analyze and report batch results"""
        total_models = len(self.results)
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        total_time = time.time() - self.start_time
        
        logger.info(f"\n{'='*80}")
        logger.info("📊 BATCH EVALUATION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"📋 Total Models: {total_models}")
        logger.info(f"✅ Successful: {len(successful)}")
        logger.info(f"❌ Failed: {len(failed)}")
        logger.info(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        
        if successful:
            # Find best performing model
            best_model = max(successful, key=lambda x: x.get('verification_roc_auc', 0))
            logger.info(f"\n🏆 Best Model: {best_model['model_name']}")
            logger.info(f"   AUC: {best_model.get('verification_roc_auc', 0):.4f}")
            logger.info(f"   TPR@FPR=1e-4: {best_model.get('verification_tpr_at_fpr_1e-4', 0):.2f}%")
            
            # Log summary to W&B
            if wandb_run:
                wandb_run.log({
                    'batch/total_models': total_models,
                    'batch/successful_models': len(successful),
                    'batch/failed_models': len(failed),
                    'batch/total_time_minutes': total_time/60,
                    'batch/best_auc': best_model.get('verification_roc_auc', 0),
                    'batch/best_tpr_fpr_1e4': best_model.get('verification_tpr_at_fpr_1e-4', 0)
                })
        
        if failed:
            logger.info(f"\n❌ Failed Models:")
            for result in failed:
                logger.info(f"   {result['model_name']}: {result.get('error', 'Unknown error')}")
        
        logger.info(f"{'='*80}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.config['result_dir'], f'evaluation_report_{timestamp}.md')
        
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']
        
        with open(report_file, 'w') as f:
            f.write("# ArcFace Batch Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset:** {self.config['target']}\n")
            f.write(f"**Total Models:** {len(self.results)}\n")
            f.write(f"**Successful:** {len(successful)}\n")
            f.write(f"**Failed:** {len(failed)}\n\n")
            
            if successful:
                f.write("## Successful Evaluations\n\n")
                
                # Create results table
                df = pd.DataFrame(successful)
                key_cols = ['model_name', 'network', 'verification_roc_auc', 'verification_tpr_at_fpr_1e-4']
                available_cols = [col for col in key_cols if col in df.columns]
                
                if available_cols:
                    table_df = df[available_cols].copy()
                    if 'verification_roc_auc' in table_df.columns:
                        table_df = table_df.sort_values('verification_roc_auc', ascending=False)
                    
                    f.write(table_df.to_markdown(index=False))
                    f.write("\n\n")
            
            if failed:
                f.write("## Failed Evaluations\n\n")
                for result in failed:
                    f.write(f"- **{result['model_name']}** ({result['network']}): {result.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            f.write("## Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2))
            f.write("\n```\n")
        
        logger.info(f"📋 Report generated: {report_file}")


def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "image_path": "/path/to/IJBC_gt_aligned",
        "result_dir": "./batch_eval_results",
        "target": "IJBC",
        "batch_size": 64,
        "num_workers": 4,
        "use_flip_test": True,
        "use_norm_score": True,
        "use_detector_score": True,
        "wandb_project": "arcface-batch-evaluation",
        "wandb_entity": null,
        "models": [
            {
                "name": "ArcFace_R100_MS1MV3",
                "network": "r100",
                "model_path": "./pretrained_models/ms1mv3_arcface_r100_fp16/backbone.pth",
                "num_features": 512,
                "timeout": 3600
            },
            {
                "name": "ArcFace_R50_MS1MV3",
                "network": "r50",
                "model_path": "./pretrained_models/ms1mv3_arcface_r50_fp16/backbone.pth",
                "num_features": 512,
                "timeout": 3600
            },
            {
                "name": "ArcFace_R34_MS1MV3",
                "network": "r34",
                "model_path": "./pretrained_models/ms1mv3_arcface_r34_fp16/backbone.pth",
                "num_features": 512,
                "timeout": 3600
            }
        ]
    }
    
    config_file = "batch_config_sample.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✅ Sample configuration created: {config_file}")
    print("📝 Edit the file with your model paths and settings, then run:")
    print(f"   python {os.path.basename(__file__)} --config {config_file}")


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate multiple ArcFace models')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--max-workers', type=int, default=1, 
                       help='Maximum number of parallel workers (default: 1)')
    parser.add_argument('--create-sample-config', action='store_true',
                       help='Create a sample configuration file and exit')
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    if not args.config:
        print("❌ Error: --config is required (use --create-sample-config to generate template)")
        return
    
    if not os.path.exists(args.config):
        print(f"❌ Error: Configuration file not found: {args.config}")
        return
    
    try:
        # Run batch evaluation
        evaluator = BatchEvaluator(args.config)
        results = evaluator.run_batch_evaluation(max_workers=args.max_workers)
        
        # Generate report
        evaluator.generate_report()
        
        logger.info(f"\n🎉 Batch evaluation completed!")
        logger.info(f"📁 Results directory: {evaluator.config['result_dir']}")
        
    except Exception as e:
        logger.error(f"❌ Batch evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()