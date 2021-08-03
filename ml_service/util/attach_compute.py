from azureml.core.workspace import Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException
from ml_service.util.env_variables import Env

def get_compute(workspace: Workspace, compute_name: str, vm_size: str, for_batch_scoring: bool = False):
    try:
        if compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and type(compute_target) is AmlCompute:
                print("Found existing compute target " + compute_name + " so using it.")
        else:
            e = Env()
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=e.vm_size,
                vm_priority=e.vm_priority,
                min_nodes=e.min_nodes,  # NOQA E501
                max_nodes=e.max_nodes,  # NOQA E501
                idle_seconds_before_scaledown="300"
            )
            compute_target = ComputeTarget.create(
                workspace,
                compute_name,
                compute_config
            )
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=10
            )
        return compute_target

    except ComputeTargetException as ex:
        print(ex)
        print("An error occurred trying to provision compute.")
        exit(1)