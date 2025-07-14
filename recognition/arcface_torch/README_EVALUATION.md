# ArcFace Comprehensive Evaluation Framework

A production-ready evaluation framework for ArcFace models on IJB-C dataset, combining the best practices from both ArcFace and CVLface approaches.

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies for evaluation
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy the example environment file
cp .env_example .env

# Edit .env with your specific paths and tokens
vim .env
```

Example `.env` configuration:
```bash
# Data Configuration
DATA_ROOT="/path/to/datasets"
IJBC_DATASET_PATH="/path/to/IJBC_gt_aligned"

# Model Configuration
MODEL_ROOT="/path/to/pretrained_models"

# Weights & Biases Configuration
WANDB_TOKEN="your_wandb_api_token"
WANDB_TEAM="your_team_name"
WANDB_PROJECT="arcface-ijbc-comprehensive"

# Optional Configuration
RESULT_DIR="./eval_results"
```

### 3. Single Model Evaluation

With environment variables configured:
```bash
# Simple usage with environment variables
python eval.py --model-path ms1mv3_arcface_r100_fp16/backbone.pth --network r100
```

Or with explicit paths:
```bash
python eval.py \
    --model-path ./pretrained_models/ms1mv3_arcface_r100_fp16/backbone.pth \
    --network r100 \
    --image-path /path/to/IJBC_gt_aligned \
    --result-dir ./eval_results \
    --wandb-project my-arcface-eval
```

### 3. Batch Evaluation

```bash
# Generate sample configuration
python batch_eval.py --create-sample-config

# Edit batch_config_sample.json with your model paths
# Run batch evaluation
python batch_eval.py --config batch_config_sample.json --max-workers 2
```

## 📁 Framework Components

```
insightface/recognition/arcface_torch/
├── eval.py                  # Main evaluation script
├── batch_eval.py            # Batch evaluation automation
├── eval_utils.py            # Utility classes and functions
├── requirements.txt         # Project dependencies
├── .env_example             # Environment variables template
└── README_EVALUATION.md     # This documentation
```

## 🎯 Features

### Comprehensive Metrics
- **Verification Metrics**: AUC, EER, TPR@FPR, Precision, Recall, F1-Score
- **Speed Metrics**: FPS, ms/image, real-time capability
- **Resource Metrics**: GPU/RAM usage, model size
- **Performance Monitoring**: Real-time memory tracking

### ArcFace Protocol Compliance
- Uses exact ArcFace preprocessing pipeline
- Supports flip test augmentation (F1/F2 modes)
- Implements norm score and detector score weighting
- Compatible with all ArcFace model architectures

### Production Ready
- Robust error handling and logging
- Parallel batch processing
- W&B integration for experiment tracking
- Comprehensive CSV and JSON output
- Automated report generation

## 📊 Evaluation Protocol

### 1. Image Preprocessing
- Face alignment using 5-point landmarks
- Resize to 112×112 pixels
- RGB normalization: [0,255] → [-1,1]
- Optional horizontal flip for test-time augmentation

### 2. Feature Extraction
- Load pre-trained ArcFace model
- Extract 512-dimensional features
- Apply feature normalization (optional)
- Weight by faceness scores (optional)

### 3. Template Aggregation
- Group images by template ID
- Aggregate features within each media
- Sum features across media in template
- L2 normalize final template features

### 4. Metric Calculation
- **Verification**: Template-to-template similarity
- **Identification**: Gallery vs probe matching
- **Performance**: Speed and resource usage

## 🔧 Configuration Options

### Single Model Evaluation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-path` | Path to backbone.pth file | Required |
| `--network` | Model architecture (r18, r34, r50, r100, etc.) | Required |
| `--image-path` | Path to IJB-C dataset directory | Required |
| `--target` | Dataset (IJBC or IJBB) | IJBC |
| `--batch-size` | Batch size for inference | 64 |
| `--num-workers` | Data loader workers | 4 |
| `--use-flip-test` | Enable flip test augmentation | True |
| `--use-norm-score` | Use norm score (TestMode N1) | True |
| `--use-detector-score` | Use detector score (TestMode D1) | True |
| `--result-dir` | Output directory | ./eval_results |
| `--wandb-project` | W&B project name | arcface-ijbc-comprehensive |

### Batch Configuration

Create a JSON configuration file:

```json
{
    "image_path": "/path/to/IJBC_gt_aligned",
    "result_dir": "./batch_eval_results",
    "target": "IJBC",
    "batch_size": 64,
    "num_workers": 4,
    "use_flip_test": true,
    "use_norm_score": true,
    "use_detector_score": true,
    "wandb_project": "arcface-batch-evaluation",
    "models": [
        {
            "name": "ArcFace_R100_MS1MV3",
            "network": "r100",
            "model_path": "./pretrained_models/ms1mv3_arcface_r100_fp16/backbone.pth",
            "num_features": 512,
            "timeout": 3600
        }
    ]
}
```

## 📋 Output Files

### Single Evaluation
- `{job}_{target}_results.csv`: Complete metrics in CSV format
- `eval_log_{timestamp}.log`: Detailed execution log
- W&B dashboard: Real-time metrics and plots

### Batch Evaluation
- `batch_results_{timestamp}.json`: Raw results for all models
- `batch_results_{timestamp}.csv`: Successful evaluations only
- `summary_{timestamp}.csv`: Sorted performance summary
- `evaluation_report_{timestamp}.md`: Comprehensive report

## 🛠️ Dataset Preparation

### IJB-C Dataset Structure
```
IJBC_gt_aligned/
├── loose_crop/              # Pre-aligned face images
│   ├── img_00001.jpg
│   ├── img_00002.jpg
│   └── ...
└── meta/                    # Metadata files
    ├── ijbc_face_tid_mid.txt        # Template-media mappings
    ├── ijbc_template_pair_label.txt # Verification pairs
    ├── ijbc_name_5pts_score.txt     # Image landmarks & scores
    ├── ijbc_1N_gallery_G1.csv      # Gallery for identification
    └── ijbc_1N_probe_mixed.csv     # Probes for identification
```

### Data Download
1. Download IJB-C dataset from official source
2. Use provided alignment tools or pre-aligned version
3. Ensure proper directory structure as shown above

## 🔬 Advanced Usage

### Custom Model Architectures

To add support for new architectures:

1. Add model definition to `backbones/`
2. Update `get_model()` function in `backbones/__init__.py`
3. Test with evaluation framework

```python
# Example: Adding custom architecture
def get_model(name, **kwargs):
    if name == "my_custom_model":
        return MyCustomModel(**kwargs)
    # ... existing code
```

### Custom Metrics

Extend `ComprehensiveMetrics` class in `eval_utils.py`:

```python
class ComprehensiveMetrics:
    @staticmethod
    def calculate_custom_metric(scores, labels):
        # Your custom metric implementation
        return {"custom_metric": value}
```

### Memory Optimization

For large-scale evaluation or limited GPU memory:

```bash
# Reduce batch size
python eval_arcface_full.py --batch-size 32 --num-workers 2

# Disable flip test to save memory
python eval_arcface_full.py --batch-size 64 --no-use-flip-test
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Reduce batch size
python eval_arcface_full.py --batch-size 16
```

**2. Model Loading Errors**
```bash
# Check model path and architecture compatibility
python -c "import torch; print(torch.load('model.pth').keys())"
```

**3. Dataset Path Issues**
```bash
# Verify dataset structure
ls -la /path/to/IJBC_gt_aligned/
ls -la /path/to/IJBC_gt_aligned/meta/
```

**4. Missing Dependencies**
```bash
# Install all requirements
pip install -r requirements.txt
```

### Performance Issues

**Slow Evaluation**
- Increase `--num-workers` for faster data loading
- Use smaller `--batch-size` if memory limited
- Enable `--use-flip-test` only if needed

**High Memory Usage**
- Monitor with system tools: `nvidia-smi`, `htop`
- Reduce batch size or disable flip test
- Check for memory leaks in custom code

### W&B Issues

**Login Problems**
```bash
wandb login your_api_key
export WANDB_API_KEY=your_api_key
```

**Disabled Logging**
```bash
# Run without W&B
python eval.py --wandb-project ""
```

## 📚 Technical Details

### Feature Extraction Pipeline

1. **Image Loading**: Load image using OpenCV
2. **Landmark Alignment**: Apply similarity transform with 5-point landmarks
3. **Preprocessing**: Resize, normalize, and optionally flip
4. **Model Inference**: Extract features using pre-trained ArcFace model
5. **Post-processing**: Normalize, weight, and aggregate features

### Metric Calculations

**Verification (1:1)**
- Calculate cosine similarity between template pairs
- Generate ROC curve and compute AUC
- Find TPR at specific FPR thresholds
- Calculate EER (Equal Error Rate)

**Identification (1:N)**
- Create similarity matrix (probes × gallery)
- Compute rank-k accuracy for k ∈ {1, 5, 10}
- Calculate identification rate at specific FAR

### Performance Monitoring

Real-time tracking of:
- GPU memory usage (via pynvml)
- System RAM usage (via psutil)
- CPU utilization
- Processing speed (images/second)

## 📄 License

This evaluation framework is part of the InsightFace project. Please refer to the main repository for license information.

## 🙏 Acknowledgments

- **InsightFace Team**: Original ArcFace implementation
- **CVLface Team**: Comprehensive evaluation metrics
- **Contributors**: Community feedback and improvements

## 📞 Support

For issues and questions:
1. Check this documentation first
2. Search existing GitHub issues
3. Create new issue with:
   - System information
   - Error logs
   - Minimal reproduction example

---

*Last updated: 2025-07-14*