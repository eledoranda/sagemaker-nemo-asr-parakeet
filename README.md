# SageMaker NeMo ASR Demo

This repository provides a sample workflow for packaging and deploying NVIDIA NeMo models on SageMaker. The example uses the Parakeet RNNT ASR model, and the same process applies to other NeMo architectures.

> **⚠️ DISCLAIMER**: This is a sample demonstration project for educational purposes only. Not intended for production use.

## Requirements

- AWS CLI configured with appropriate permissions
- Python 3.11+

## Quick Start

```bash
# Use Python 3.11 
# Install dependencies
pip install -r requirements.txt

# Deploy model
python deploy.py

# Test endpoint
python test.py
```

## Architecture

- **Model**: NVIDIA Parakeet RNNT 0.6B
- **Platform**: SageMaker PyTorch 2.4
- **Instance**: ml.g5.xlarge
- **Input**: 16kHz mono PCM WAV (base64)
- **Output**: JSON transcript

## Project Structure

```
├── deploy.py           # Main deployment script
├── test.py            # Client testing
├── model/
│   └── inference.py   # SageMaker handler
├── utils/
│   ├── prepare_nemo_model.py
│   └── create_role.py
└── artifacts/         # Model storage
```

