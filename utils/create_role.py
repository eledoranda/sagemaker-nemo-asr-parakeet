"""
utils/create_role.py — SageMaker Execution Role Utility
-------------------------------------------------------
This helper ensures a SageMaker execution role exists for deployments.
If the role does not exist, it is created with a trust policy for SageMaker
and attached with the `AmazonSageMakerFullAccess` managed policy.

Steps:
  1. Define a trust policy allowing SageMaker to assume the role.
  2. Attempt to create the role.
  3. Attach required managed policy.
  4. If the role already exists, return its ARN instead.

Returns:
  • str: The ARN of the SageMaker execution role.
"""

import boto3
import json
import logging

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def create_sagemaker_role(role_name: str = "SageMakerExecutionRole-Parakeet") -> str:
    """
    Ensure a SageMaker execution role exists, creating it if necessary.

    Args:
        role_name: Name of the IAM role to create or retrieve.

    Returns:
        str: The ARN of the SageMaker execution role.

    Raises:
        RuntimeError: If role creation or retrieval fails.
    """
    iam = boto3.client("iam")

    # Trust policy: allow SageMaker service to assume this role
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        # Attempt to create role
        log.info(f"🔧 Creating IAM role: {role_name}")
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="SageMaker execution role for Parakeet model",
        )

        # Attach SageMaker managed policy
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        )

        role_arn = response["Role"]["Arn"]
        log.info(f"✅ Created new role: {role_arn}")
        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        # Role already exists → fetch ARN
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        log.info(f"ℹ️ Role already exists: {role_arn}")
        return role_arn

    except Exception as e:
        raise RuntimeError(f"Failed to create or retrieve SageMaker role '{role_name}': {e}") from e


# ---------------------------------------------------------------------
# CLI Usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    arn = create_sagemaker_role()
    print(f"SageMaker Execution Role ARN: {arn}")
