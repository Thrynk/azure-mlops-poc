"""
Env dataclass to load and hold all environment variables
"""
from dataclasses import dataclass
import os
from typing import Optional

from dotenv import load_dotenv

@dataclass(frozen=True)
class Env:
    """Loads all environment variables into a predefined set of properties
    """

    # to load .env file into environment variables for local execution
    load_dotenv()
    workspace_name: Optional[str] = os.environ.get("AML_WORKSPACE_NAME")
    resource_group: Optional[str] = os.environ.get("RESOURCE_GROUP")
    subscription_id: Optional[str] = os.environ.get("SUBSCRIPTION_ID")
    sources_directory_train: Optional[str] = os.environ.get(
        "SOURCES_DIR_TRAIN"
    )
    train_script_path: Optional[str] = os.environ.get("TRAIN_SCRIPT_PATH")
    evaluate_script_path: Optional[str] = os.environ.get(
        "EVALUATE_SCRIPT_PATH"
    )  # NOQA: E501
    register_script_path: Optional[str] = os.environ.get(
        "REGISTER_SCRIPT_PATH"
    )  # NOQA: E501
    datastore_name: Optional[str] = os.environ.get("DATASTORE_NAME")
    model_name: Optional[str] = os.environ.get("MODEL_NAME")
    dataset_version: Optional[str] = os.environ.get("DATASET_VERSION")
    dataset_name: Optional[str] = os.environ.get("DATASET_NAME")
    vm_size: Optional[str] = os.environ.get("AML_COMPUTE_CLUSTER_CPU_SKU")
    compute_name: Optional[str] = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    vm_priority: Optional[str] = os.environ.get(
        "AML_CLUSTER_PRIORITY", "lowpriority"
    )
    min_nodes: int = int(os.environ.get("AML_CLUSTER_MIN_NODES", 0))
    max_nodes: int = int(os.environ.get("AML_CLUSTER_MAX_NODES", 4))
    aml_env_name: Optional[str] = os.environ.get("AML_ENV_NAME")
    aml_env_train_conda_dep_file: Optional[str] = os.environ.get(
        "AML_ENV_TRAIN_CONDA_DEP_FILE", "conda_dependencies.yml"
    )
    rebuild_env: Optional[bool] = os.environ.get(
        "AML_REBUILD_ENVIRONMENT", "false"
    ).lower().strip() == "true"

    run_evaluation: Optional[str] = os.environ.get("RUN_EVALUATION", "true")
    pipeline_name: Optional[str] = os.environ.get("TRAINING_PIPELINE_NAME")
    build_id: Optional[str] = os.environ.get("BUILD_BUILDID")