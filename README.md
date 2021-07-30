# mlops-poc-train

## Introduction

### Context

This repository is intended to reproduce the [MLOps Python repository](https://github.com/microsoft/MLOpsPython/) of Azure.

I wanted to reproduce step by step, the project to understand better the different tools to use, and the overall concept of integrating MLOps in a Machine Learning Project in Microsoft Azure.
And also to add some custom tools, that I think are useful for MLOps.

### A few words on MLOps Python project

> This reference architecture shows how to implement continuous integration (CI), continuous delivery (CD), and retraining pipeline for an AI application using Azure DevOps and Azure Machine Learning. The solution is built on the scikit-learn diabetes dataset but can be easily adapted for any AI scenario and other popular build systems such as Jenkins or Travis.

You can find the documentation for this reference architecture [here](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/mlops-python).

This architecture respects Azure best practices and help you to have a solid architecture base for your own project.

I will allow myself to add some modifications on the architecture but they will be mentioned further (no worry :wink:)

## Architecture

### Overview

This architecture consists of the following components:

> **Azure Pipelines**. This build and test system is based on Azure DevOps and used for the build and release pipelines. Azure Pipelines breaks these pipelines into logical steps called tasks. For example, the Azure CLI task makes it easier to work with Azure resources.
>
> **Azure Machine Learning** is a cloud service for training, scoring, deploying, and managing machine learning models at scale. This architecture uses the Azure Machine Learning Python SDK to create a workspace, compute resources, the machine learning pipeline, and the scoring image. An Azure Machine Learning workspace provides the space in which to experiment, train, and deploy machine learning models.
>
> **Azure Machine Learning Compute** is a cluster of virtual machines on-demand with automatic scaling and GPU and CPU node options. The training job is executed on this cluster.
>
> **Azure Machine Learning pipelines** provide reusable machine learning workflows that can be reused across scenarios. Training, model evaluation, model registration, and image creation occur in distinct steps within these pipelines for this use case. The pipeline is published or updated at the end of the build phase and gets triggered on new data arrival.
>
> **Azure Blob Storage**. Blob containers are used to store the logs from the scoring service. In this case, both the input data and the model prediction are collected. After some transformation, these logs can be used for model retraining.
>
> **Azure Container Registry**. The scoring Python script is packaged as a Docker image and versioned in the registry.
>
> **Azure Container Instances**. As part of the release pipeline, the QA and staging environment is mimicked by deploying the scoring webservice image to Container Instances, which provides an easy, serverless way to run a container.
>
> **Azure Kubernetes Service**. Once the scoring webservice image is thoroughly tested in the QA environment, it is deployed to the production environment on a managed Kubernetes cluster.
>
> **Azure Application Insights**. This monitoring service is used to detect performance anomalies.

Source : <https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/mlops-python>

### Environment setup

For this project, we need different Azure resources :

- an Azure Blob Storage
- a Key Vault
- an Application Insights
- a Container Registry
- an Azure Machine Learning workspace

#### ***Infrastructure as code (ARM templates)***

To create all the resources, we will use the Azure Resource Manager templates (ARM templates) to have an infrastructure as code. [Here](https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/overview) is the overview of what is ARM templates.

This allows us to control our Azure resources more formally between developers and data scientists.

We will create the templates under the [environment_setup](/environment_setup) folder. You can find more info in the documentation on the file to understand which resources are set up and why.

We will have :

- a [main.bicep](/environment_setup/src/main.bicep) file on which will create the resources with the Azure Resource Manager. Azure Resource Manager now handles bicep compilation for you.
It will be deployed like this in the Azure DevOps Pipeline :

```bash
az deployment group create --template-file ./environment_setup/src/main.bicep -g mlops-AML-RG --parameters ./environment_setup/src/parameters.json
```

You can check the impact of the deployment with :

```bash
az deployment group what-if --template-file ./environment_setup/src/main.bicep -g mlops-train-AML-RG --parameters ./environment_setup/src/parameters.json
```

- a [iac-create-environment-pipeline-arm.yml](/environment_setup/pipelines/iac-create-environment-pipeline-arm.yml) file that is an Azure pipeline (in Azure DevOps), to help automate the process of deploying resources each time there is a change on the templates pushed to the repository (CI/CD principle).

We then need to choose our development platform.

#### ***Development platform (Local or on Compute instance in Azure Machine Learning in the development resource group)***

We need to set up a development environment that allows us to share and reproduce experiments.

A good practice is to run your experiments using Jupyter notebooks, but what can we create to allow Data Scientists to share their experiments and reproduce experiments done by other Data Scientists ?

First, we can share our code using git (allows versioning and centralized code for the whole project) and Azure Machine Learning studio to have shared files and experiments.

We have 2 choices :

**Web interface** :

You can either code your experiments in the JupyterLab of a Compute Instance on the ml.azure.com portal.
This Compute instance is linked to your workspace :

- You can code your in the JupyterLab with the Notebooks tab.
- manage :
    1. your datasets.
    2. your experiments.
    3. your pipelines
    4. your models (with versioning).
    5. your deployment points.
    6. your compute instances.
    7. your environments using the web interface.
    8. your datastores.
    9. push to the Git repo with the terminal of your compute instance.

You can learn more on how to run Jupyter notebooks [here](https://docs.microsoft.com/fr-fr/azure/machine-learning/how-to-run-jupyter-notebooks).

**Visual studio code** :

Or you can use visual studio code if you like to have a version of the code on your laptop and tune your development IDE.
This also allows you to use your own laptop compute power if you prefer to reduce costs for your experiments.

Installation for VS code:

1. Install VS code
2. Install Azure Machine Learning extension
3. Connect to your Azure Account.

You now have the same functionalities than on the web interface in your Visual Studio code environment.

If you want the full setup tutorial, check [here](https://docs.microsoft.com/fr-fr/azure/machine-learning/how-to-setup-vs-code).

#### ***Production environment***

TO DO

### Best practices to integrate MLOps in ML lifecycle in Azure

Here are the best practices we chose to follow (:warning: I'm not saying you cannot have the best practices of MLOps in your project if you don't use this principles, for example you can reproduce a lot of this concepts using Kubeflow on an Azure Kubernetes Cluster, but this configuration needs to maintain an AKS cluster and then was not chosen for my project, so it is out of the scope of this repository, check Google Cloud MLOps course on Coursera if you want to know more).

#### **Experimenting**

##### **Tracking experiments using MLFlow**

We are going to use MLflow to track our different experiments. As Data Scientists, we experiment in Notebooks. The idea here is to provide a reproductible experiment with the different parameters, models, results and notebook versions.
First, create a Notebook corresponding to your experiment in the [experimentation](/experimentation) folder.

In a traditional way, the notebook would look like [this](/experimentation/Diabetes%20Ridge%20Regression%20Training.ipynb). (If you want to execute this notebook, you need to be familiar with the [Experimenting : environments](#environments) part to setup the needed environment for the project).

When using MLflow, you need to have a notebook that is like [this](/experimentation/Diabetes%20Ridge%20Regression%20Training%20MLflow.ipynb) to track your experiment so that it can appear in Azure.

- tracking parameters :

  - It is important to track your parameters when experimenting to allow future data scientists to know which parameters have been tried in the past.
- tracking models :

  - Versioning your models is important to compare them and keep track of which model is better.
- tracking results (metrics) :

  - tracking your metrics allows to know how your model performed in the run and eventually if it is better than another one according to these particular metrics.

##### **Versioning (Git)**

Versioning the code is very important because it allows Data Scientists to know who did what so they can reach out the person if they have a question.

It also allows to track different versions of the code, to share and rollback to a previous version.

Commits represent an history of the project. So just reading all commits, you should be able to understand the history of the project.

You can learn more on  [Git](https://git-scm.com/about) to understand better all the features.

But our main interest here in MLOps, is to track code version and who did what.

This tool is very helpful to implement pipelines with Azure DevOps.

##### **Sharing in Workspace**

You can eventually share your work in the shared workspace of Azure Machine Learning, so that you can access quickly your notebooks in the Web UI of Azure Machine Learning Workspace.

But this is not replacing Git (which is our priority) because it doesn't allow version control.

##### **Using versioned datasets**

You should create Datasets using your datastore storage (workspaceblobstorage is the default datastore of your workspace).
If you take the example of a csv file you want to upload and create a dataset on. The file will be uploaded to your datastore (workspaceblobstorage).

You can manage different versions of your datasets and also see which models were trained on this particular dataset.

Then, you can consume your dataset in your code.

##### **Documenting your experiments**

It is important to document your experiments and answer :

- What experiment are you trying to do ?
- Why are you doing this experiment ?

You can answer those 2 questions each time you write a complex peace of code.

##### **Environments**

[Azure Machine Learning Environments](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments) are important for reproductability of your experiments and also to productionalize your models.

You need these environments to submit a run to Azure Machine Learning Compute instances.

How to setup a local environment to run your experiments :

```bash
conda env create -f diabetes_regression/conda_dependencies.yml
```

To run this environment in Azure ML, this yaml file need to be in the source directory you will provide (here [diabetes_regression](/diabetes_regression) folder). 

The source directory is a needed argument for a [PythonScriptStep](https://docs.microsoft.com/fr-fr/python/api/azureml-pipeline-steps/azureml.pipeline.steps.python_script_step.pythonscriptstep?view=azure-ml-py) in a pipeline).

We will see what is a PythonScriptStep when creating pipelines.

#### **Go to a 1st version of production-ready code**

Then when you are satisfied with your experiment results, you might want to go to production with your model. For that you need to automate training and deployment of your model. When you push on the main branch or a release branch, you want your model to be trained, all artifacts to be saved in your workspace, all results to be tracked and your model to be evaluated against current production model.

To do that, you need to create a more production ready-code of your experiment, that would allow you with just a few scripts and command line to train, evaluate and deploy your model.

This work is in the [diabetes_regression](/diabetes_regression) folder.

The idea is to create functions for each step of your workflow : training, evaluating your model, register your model and score an incoming example.

What we need to do is refactor our code [here](/experimentation/Diabetes%20Ridge%20Regression%20Training%20MLflow.ipynb) into functions :

1. Create a function called split_data to split dataframe into training and test.

2. Create a function called train_model which takes data and args (parameters for model), and returns a trained model.

3. Create a function called get_model_metrics with parameters : reg_model and data taht evaluates the model on test data and return a dict of metrics. (that can be logged with MLFlow or with log function of Run object from azureml.core).

You can find this script [here](/diabetes_regression/training/train.py).

Then we need a script that uses these functions and run them in our workspace and on wanted dataset version. This script will also be used as a step in our future pipeline for training the model. We also need to log the parameters and metrics to the AML Workspace.

This script is [here](/diabetes_regression/training/train_aml.py).

#### **Write pipelines**
