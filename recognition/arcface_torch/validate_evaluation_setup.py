#!/usr/bin/env python3
"""
Validation Script for ArcFace Evaluation Framework
Checks dependencies, dataset structure, and model availability before running evaluation.

Usage:
    python validate_evaluation_setup.py --image-path /path/to/IJBC_gt_aligned
    python validate_evaluation_setup.py --check-model ./pretrained_models/backbone.pth --network r100
"""

import os
import sys
import argparse
import importlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidationResult:
    def __init__(self, name: str, passed: bool, message: str, critical: bool = True):
        self.name = name
        self.passed = passed
        self.message = message
        self.critical = critical


class EvaluationValidator:
    """Comprehensive validation for ArcFace evaluation setup"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            ('torch', 'PyTorch for deep learning'),
            ('torchvision', 'PyTorch vision utilities'),
            ('numpy', 'Numerical computing'),
            ('pandas', 'Data manipulation'),
            ('scikit-learn', 'Machine learning metrics'),
            ('cv2', 'OpenCV for image processing'),
            ('skimage', 'Scikit-image for transformations'),
            ('tqdm', 'Progress bars'),
            ('psutil', 'System monitoring'),
        ]
        
        optional_packages = [
            ('pynvml', 'GPU memory monitoring'),
            ('wandb', 'Experiment tracking'),
            ('menpo', 'Face analysis (for IJB evaluation)'),
        ]
        
        all_passed = True
        
        # Check required packages
        for package, description in required_packages:
            try:
                importlib.import_module(package)
                self.results.append(ValidationResult(
                    f"Required: {package}", True, f"✅ {description}", critical=True
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    f"Required: {package}", False, f"❌ Missing: {description}", critical=True
                ))
                all_passed = False
        
        # Check optional packages
        for package, description in optional_packages:
            try:
                importlib.import_module(package)
                self.results.append(ValidationResult(
                    f"Optional: {package}", True, f"✅ {description}", critical=False
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    f"Optional: {package}", False, f"⚠️  Missing: {description} (optional)", critical=False
                ))
        
        return all_passed
    
    def check_evaluation_files(self) -> bool:
        """Check if evaluation framework files exist"""
        logger.info("🔍 Checking evaluation framework files...")
        
        required_files = [
            ('eval_utils.py', 'Evaluation utilities'),
            ('eval_arcface_full.py', 'Main evaluation script'),
            ('batch_eval_arcface.py', 'Batch evaluation script'),
            ('requirements_eval.txt', 'Additional requirements'),
            ('README_EVALUATION.md', 'Evaluation documentation'),
        ]
        
        all_passed = True
        
        for filename, description in required_files:
            file_path = Path(__file__).parent / filename
            if file_path.exists():
                self.results.append(ValidationResult(
                    f"Framework: {filename}", True, f"✅ {description}", critical=True
                ))
            else:
                self.results.append(ValidationResult(
                    f"Framework: {filename}", False, f"❌ Missing: {description}", critical=True
                ))
                all_passed = False
        
        return all_passed
    
    def check_dataset_structure(self, image_path: str) -> bool:
        """Check IJB-C dataset structure"""
        logger.info(f"🔍 Checking dataset structure: {image_path}")
        
        if not os.path.exists(image_path):
            self.results.append(ValidationResult(
                "Dataset: Path", False, f"❌ Dataset path does not exist: {image_path}", critical=True
            ))
            return False
        
        # Check main directories
        required_dirs = [
            ('loose_crop', 'Pre-aligned face images'),
            ('meta', 'Metadata files'),
        ]
        
        dirs_passed = True
        for dirname, description in required_dirs:
            dir_path = os.path.join(image_path, dirname)
            if os.path.exists(dir_path):
                self.results.append(ValidationResult(
                    f"Dataset: {dirname}/", True, f"✅ {description}", critical=True
                ))
            else:
                self.results.append(ValidationResult(
                    f"Dataset: {dirname}/", False, f"❌ Missing: {description}", critical=True
                ))
                dirs_passed = False
        
        if not dirs_passed:
            return False
        
        # Check metadata files
        meta_dir = os.path.join(image_path, 'meta')
        required_meta_files = [
            ('ijbc_face_tid_mid.txt', 'Template-media mappings'),
            ('ijbc_template_pair_label.txt', 'Verification pairs'),
            ('ijbc_name_5pts_score.txt', 'Image landmarks and scores'),
        ]
        
        meta_passed = True
        for filename, description in required_meta_files:
            file_path = os.path.join(meta_dir, filename)
            if os.path.exists(file_path):
                # Check file size
                file_size = os.path.getsize(file_path)
                if file_size > 1000:  # Reasonable minimum size
                    self.results.append(ValidationResult(
                        f"Metadata: {filename}", True, f"✅ {description} ({file_size:,} bytes)", critical=True
                    ))
                else:
                    self.results.append(ValidationResult(
                        f"Metadata: {filename}", False, f"⚠️  {description} seems too small ({file_size} bytes)", critical=True
                    ))
                    meta_passed = False
            else:
                self.results.append(ValidationResult(
                    f"Metadata: {filename}", False, f"❌ Missing: {description}", critical=True
                ))
                meta_passed = False
        
        # Check sample images
        loose_crop_dir = os.path.join(image_path, 'loose_crop')
        try:
            image_files = list(Path(loose_crop_dir).glob('*.jpg'))
            if len(image_files) > 100:  # Should have many images
                self.results.append(ValidationResult(
                    "Dataset: Images", True, f"✅ Found {len(image_files):,} image files", critical=True
                ))
            else:
                self.results.append(ValidationResult(
                    "Dataset: Images", False, f"⚠️  Only found {len(image_files)} images (expected thousands)", critical=True
                ))
                meta_passed = False
        except Exception as e:
            self.results.append(ValidationResult(
                "Dataset: Images", False, f"❌ Error counting images: {e}", critical=True
            ))
            meta_passed = False
        
        return meta_passed
    
    def check_model_file(self, model_path: str, network: str = 'r100') -> bool:
        """Check if model file is valid and loadable"""
        logger.info(f"🔍 Checking model: {model_path}")
        
        if not os.path.exists(model_path):
            self.results.append(ValidationResult(
                "Model: File", False, f"❌ Model file does not exist: {model_path}", critical=True
            ))
            return False
        
        # Check file size (models should be reasonably large)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # Less than 1MB is suspicious
            self.results.append(ValidationResult(
                "Model: Size", False, f"⚠️  Model file seems too small: {file_size:,} bytes", critical=True
            ))
            return False
        else:
            self.results.append(ValidationResult(
                "Model: Size", True, f"✅ Model file size: {file_size / (1024**2):.1f} MB", critical=True
            ))
        
        # Try to load model
        try:
            import torch
            from backbones import get_model
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            self.results.append(ValidationResult(
                "Model: Load", True, f"✅ Successfully loaded state dict with {len(state_dict)} keys", critical=True
            ))
            
            # Try to create model architecture
            model = get_model(network, dropout=0, fp16=False, num_features=512)
            self.results.append(ValidationResult(
                "Model: Architecture", True, f"✅ Successfully created {network} architecture", critical=True
            ))
            
            # Try to load weights
            model.load_state_dict(state_dict)
            self.results.append(ValidationResult(
                "Model: Weights", True, f"✅ Successfully loaded weights into {network} model", critical=True
            ))
            
            return True
            
        except Exception as e:
            self.results.append(ValidationResult(
                "Model: Load", False, f"❌ Error loading model: {e}", critical=True
            ))
            return False
    
    def check_gpu_availability(self) -> bool:
        """Check GPU availability and memory"""
        logger.info("🔍 Checking GPU availability...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.results.append(ValidationResult(
                    "GPU: Availability", True, f"✅ CUDA available with {gpu_count} GPU(s)", critical=False
                ))
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    self.results.append(ValidationResult(
                        f"GPU {i}: {gpu_name}", True, f"✅ {gpu_memory:.1f} GB memory", critical=False
                    ))
                
                return True
            else:
                self.results.append(ValidationResult(
                    "GPU: Availability", False, "⚠️  CUDA not available, will use CPU (slow)", critical=False
                ))
                return False
                
        except Exception as e:
            self.results.append(ValidationResult(
                "GPU: Check", False, f"❌ Error checking GPU: {e}", critical=False
            ))
            return False
    
    def run_validation(self, image_path: Optional[str] = None, 
                      model_path: Optional[str] = None, 
                      network: str = 'r100') -> bool:
        """Run comprehensive validation"""
        logger.info("🚀 Starting ArcFace Evaluation Framework Validation")
        logger.info("=" * 70)
        
        # Run all checks
        deps_ok = self.check_dependencies()
        files_ok = self.check_evaluation_files()
        gpu_ok = self.check_gpu_availability()
        
        dataset_ok = True
        if image_path:
            dataset_ok = self.check_dataset_structure(image_path)
        
        model_ok = True
        if model_path:
            model_ok = self.check_model_file(model_path, network)
        
        # Print results
        self._print_results()
        
        # Overall assessment
        critical_failed = any(not r.passed for r in self.results if r.critical)
        
        logger.info("\n" + "=" * 70)
        if critical_failed:
            logger.error("❌ Validation FAILED - Critical issues found")
            return False
        else:
            logger.info("✅ Validation PASSED - Setup looks good!")
            return True
    
    def _print_results(self):
        """Print validation results in organized format"""
        logger.info("\n📊 Validation Results:")
        logger.info("-" * 70)
        
        # Group results by category
        categories = {}
        for result in self.results:
            category = result.name.split(':')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Print by category
        for category, results in categories.items():
            logger.info(f"\n{category}:")
            for result in results:
                status = "✅" if result.passed else ("❌" if result.critical else "⚠️ ")
                logger.info(f"  {status} {result.message}")
    
    def generate_setup_commands(self):
        """Generate helpful setup commands"""
        logger.info("\n🔧 Setup Commands:")
        logger.info("-" * 70)
        
        missing_required = [r for r in self.results if not r.passed and r.critical and 'Required:' in r.name]
        if missing_required:
            logger.info("\n📦 Install missing dependencies:")
            logger.info("pip install -r requirements_eval.txt")
        
        dataset_issues = [r for r in self.results if not r.passed and 'Dataset:' in r.name]
        if dataset_issues:
            logger.info("\n📁 Dataset setup:")
            logger.info("1. Download IJB-C dataset from official source")
            logger.info("2. Ensure proper directory structure:")
            logger.info("   IJBC_gt_aligned/")
            logger.info("   ├── loose_crop/     # Pre-aligned images")
            logger.info("   └── meta/           # Metadata files")
        
        model_issues = [r for r in self.results if not r.passed and 'Model:' in r.name]
        if model_issues:
            logger.info("\n🤖 Model setup:")
            logger.info("1. Download pre-trained ArcFace models")
            logger.info("2. Ensure backbone.pth files are accessible")
            logger.info("3. Verify model architecture matches network parameter")


def main():
    parser = argparse.ArgumentParser(description='Validate ArcFace evaluation setup')
    parser.add_argument('--image-path', type=str, help='Path to IJB-C dataset directory')
    parser.add_argument('--check-model', type=str, help='Path to model file to validate')
    parser.add_argument('--network', type=str, default='r100', help='Model architecture (r100, r50, etc.)')
    parser.add_argument('--generate-sample-config', action='store_true', help='Generate sample batch config')
    
    args = parser.parse_args()
    
    if args.generate_sample_config:
        # Import and run batch config generator
        try:
            from batch_eval_arcface import create_sample_config
            create_sample_config()
        except ImportError:
            logger.error("❌ batch_eval_arcface.py not found")
        return
    
    # Run validation
    validator = EvaluationValidator()
    success = validator.run_validation(
        image_path=args.image_path,
        model_path=args.check_model,
        network=args.network
    )
    
    # Generate setup commands if there were issues
    if not success:
        validator.generate_setup_commands()
    else:
        logger.info("\n🎉 Ready to run ArcFace evaluation!")
        logger.info("\nNext steps:")
        logger.info("1. Single evaluation:")
        logger.info("   python eval_arcface_full.py --model-path MODEL --network ARCH --image-path DATA")
        logger.info("2. Batch evaluation:")
        logger.info("   python validate_evaluation_setup.py --generate-sample-config")
        logger.info("   python batch_eval_arcface.py --config batch_config_sample.json")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()