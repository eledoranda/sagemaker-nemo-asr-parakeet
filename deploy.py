"""
deploy.py — Minimal SageMaker deployment script
----------------------------------------------
This script automates the deployment of a NeMo ASR model to Amazon SageMaker.
It performs the following steps:
  1. Prepare a model artifact (download → package → tar.gz).
  2. Configure AWS resources (session, role, bucket).
  3. Upload the model artifact to Amazon S3.
  4. Validate the inference handler code.
  5. Create a SageMaker model definition.
  6. Deploy the model (create or update an endpoint).

Note: This is sample code provided for demonstration purposes.
"""

import os
import logging
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from utils.prepare_nemo_model import prepare_nemo_artifact
from utils.create_role import create_sagemaker_role

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Step 1: Prepare model artifact
# ---------------------------------------------------------------------
log.info("📦 Step 1: Preparing model artifact...")
tar_path = prepare_nemo_artifact(
    local_nemo_path="artifacts/model.nemo",
    model_name="nvidia/parakeet-rnnt-0.6b",
    out_tar="artifacts/model.tar.gz",
)
log.info(f"✅ Model artifact created at: {tar_path}")

# ---------------------------------------------------------------------
# Step 2: Configure AWS session, IAM role, and S3 bucket
# ---------------------------------------------------------------------
log.info("🔧 Step 2: Setting up AWS session and resources...")
boto_sess = boto3.Session()
sess = sagemaker.Session(boto_session=boto_sess)
role = create_sagemaker_role()
bucket = sess.default_bucket()
prefix = "nemo-parakeet"

log.info(f"✅ Using S3 bucket: {bucket}")
log.info(f"✅ Using IAM role: {role}")

# ---------------------------------------------------------------------
# Step 3: Upload model artifact to Amazon S3
# ---------------------------------------------------------------------
log.info("☁️ Step 3: Uploading model artifact to S3...")
model_s3 = sess.upload_data(tar_path, bucket=bucket, key_prefix=prefix)
log.info(f"✅ Model uploaded to: {model_s3}")

# ---------------------------------------------------------------------
# Step 4: Validate inference code
# ---------------------------------------------------------------------
log.info("📁 Step 4: Validating inference handler code...")
code_dir = "model"  # must contain inference.py that loads /opt/ml/model/model.nemo
assert os.path.exists(os.path.join(code_dir, "inference.py")), \
    "❌ Missing inference.py in 'model/' directory"
log.info(f"✅ Inference code found in: {code_dir}")

# ---------------------------------------------------------------------
# Step 5: Define SageMaker model
# ---------------------------------------------------------------------
log.info("🏗️ Step 5: Creating SageMaker model definition...")
sm_model = PyTorchModel(
    model_data=model_s3,
    role=role,
    framework_version="2.4",
    py_version="py311",
    entry_point="inference.py",
    source_dir=code_dir,
    sagemaker_session=sess,
)
log.info("✅ SageMaker model definition created")

# ---------------------------------------------------------------------
# Step 6: Deploy model to endpoint
# ---------------------------------------------------------------------
endpoint_name = "nemo-parakeet-demo"
log.info(f"🚀 Step 6: Creating new endpoint: {endpoint_name}")
predictor = sm_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name=endpoint_name,
    wait=True,
)
log.info(f"🎉 Endpoint deployed: {predictor.endpoint_name}")
