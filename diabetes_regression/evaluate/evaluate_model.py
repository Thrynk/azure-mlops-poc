from hashlib import new
import traceback
import argparse
from azureml.core import Run
from azureml.core.workspace import Workspace
from azureml.core.model import Model as AMLModel

def get_current_workspace() -> Workspace:
    """
    Retrieves and returns the current workspace.
    Will not work when ran locally.

    Parameters:
    None

    Return:
    The current workspace.
    """
    run = Run.get_context(allow_offline=False)
    experiment = run.experiment
    return experiment.workspace

def get_model(
    model_name: str,
    model_version: int = None, # If none, return latest model
    tag_name: str = None,
    tag_value: str = None,
    aml_workspace : Workspace = None
) -> AMLModel:
    """ Retrieves and returns a model from the workspace by its name
    and (optional) tag.

    Parameters:
    aml_workspace (Workspace): aml.core Workspace that the model lives.
    model_name (str): name of the model we are looking for
    (optional) model_version (str): model version. Latest if not provided.
    (optional) tag (str): the tag value & name the model was registered under.

    Return:
    A single aml model from the workspace that matches the name and tag, or
    None.
    """
    if aml_workspace is None:
        print("No workspace defined - using current experiment workspace.")
        aml_workspace = get_current_workspace()

    tags = None
    if tag_name is not None or tag_value is not None:
        # Both a name and value must be specified to use tags.
        if tag_name is None or tag_value is None:
            raise ValueError(
                "model_tag_name and model_tag_value should both be supplied or excluded"
            )
        
        tags = [[tag_name, tag_value]]
    
    model = None
    if model_version is not None:
        model = AMLModel(
            aml_workspace,
            name=model_name,
            version=model_version,
            tags=tags
        )
    else:
        models = AMLModel.list(
            aml_workspace,
            name= model_name,
            tags=tags,
            latest=True
        )
        if len(models) == 1:
            model = models[0]
        elif len(models) > 1:
            raise Exception("Expected only one model")

    return model

run = Run.get_context()
experiment = run.experiment
ws = run.experiment.workspace
run_id = 'amlcompute'

parser = argparse.ArgumentParser("evaluate")

parser.add_argument(
    "--run_id",
    type=str,
    help="Training run ID"
)

parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the model in production",
    default="diabetes_model.pkl"
)

parser.add_argument(
    "--allow_run_cancel",
    type=str,
    help="Set this value to false to avoid evaluation step from cancelling run after an unsuccessful evaluation",
    default="true"
)

args = parser.parse_args()
if(args.run_id is not None):
    run_id = args.run_id
if(run_id == 'amlcompute'):
    run_id = run.parent.id

model_name = args.model_name
metric_eval = "mse"

allow_run_cancel = args.allow_run_cancel

# Parameterize the matrices on which the models should be compared
# Add golden data set on which all the model performance can be evaluated

try:
    tag_name = 'experiment_name'

    model = get_model(
        model_name=model_name,
        tag_name=tag_name,
        tag_value=experiment.name,
        aml_workspace=ws
    )

    if (model is not None):
        production_model_mse = 10000
        if (metric_eval in model.tags):
            production_model_mse = float(model.tags[metric_eval])
        new_model_mse = float(run.parent.get_metrics().get(metric_eval))

        if(production_model_mse is None or new_model_mse is None):
            print("Unable to find", metric_eval, "metrics, existing evaluation")

            if(allow_run_cancel.lower() == 'true'):
                run.parent.cancel()

        else:
            print(
                "Current Production model mse: {}, "
                "New trained model mse: {}".format(
                    production_model_mse, new_model_mse
                )
            )

        if (new_model_mse < production_model_mse):
            print("New trained model performs better, "
                  "thus it should be registered")
        else:
            print("New trained model metric is worse than or equal to "
                  "production model so skipping model registration.")
            if((allow_run_cancel).lower() == 'true'):
                run.parent.cancel()
    else:
        print("This is the first model, "
              "thus it should be registered")

except Exception :
    traceback.print_exc(limit=None, file=None, chain=True)
    print("Something went wrong trying to evaluate. Exiting.")
    raise 