# Hyperdrive vs AutoML + Deployment in AzureML Studio

## Table of Contents
   * [Project Set Up and Installation](#Project-Set-Up-and-Installation)
   * [Dataset](#Dataset)
   * [Overview](#Overview)
   * [Task](#Task)
   * [Access](#Access)
   * [Automated ML](#Automated-ML)
   * [AutoML-Results](#AutoML-Results)
   * [Hyperparameter-Tuning](#Hyperparameter-Tuning)
   * [Hyperparameter-Tuning-Results](#Hyperparameter-Tuning-Results)
   * [Model Deployment](#Model-Deployment)
   * [Screen Recording](#Screen-Recording)
   * [Future Work](#Future-Work)



In this project, we will train a model using HyperDrive to search within a space of hyperparameters for a Random Forest model. We will compare the resulting model with the best model obtained via AutoML. Instead of searching a hyperparameter space, AutoML will test a variety of models with some automatically chosen hyperparameters and see which one works best.

We will be working with the [Pump it Up Dataset](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/). We've modified the dataset so that it works better in Azure (we concatenated y_test to X_test).

After we've trained the models, we will deploy the model that performs the best on `recall_score_micro`.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

To use this repository, you must import the dataset in AzureML Studio in Datasets, and then import the code in the Notebooks section.

To load a dataset in AzureML Studio, you first must follow these steps:

1. Go to the Datasets page.

![import_dataset_1.png](./imgs/import_dataset_1.png)

2. Click on Create dataset and the appropriate drop-down option. In our case, we're loading the dataset locally. The dataset (which was modified from the DrivenData competition) can be found [here](https://github.com/JayThibs/hyperdrive-vs-automl-plus-deployment/blob/main/Pump-it-Up-dataset.csv).

![import_dataset_2.png](./imgs/import_dataset_2.png)

You will then be asked to give your dataset a name and specify the dataset type. Our dataset is a tabular dataset. Afterwards, click next.

![import_dataset_3.png](./imgs/import_dataset_3.png)

Now, we import the dataset with the Browse button and AzureML Studio will make sure there is no issue with the dataset.

![import_dataset_4.png](./imgs/import_dataset_4.png)
![import_dataset_5.png](./imgs/import_dataset_5.png)

We will now make sure that dataset is being loaded with the correct encoding, delimiters, etc. For our case, we change the Column Headers so that it uses the headers that came with the file (instead of putting the headers as a first row).

![import_dataset_6.png](./imgs/import_dataset_6.png)

Make sure to check that all the columns have the correct type. We don't change anything here.

![import_dataset_7.png](./imgs/import_dataset_7.png)

Click next and your dataset will be loaded into AzureML Studio!

![import_dataset_8.png](./imgs/import_dataset_8.png)

Lastly, we go to the Notebooks page and load the coad from this repository. Click Create and load the folder.

![import_code_repo.png](./imgs/import_code_repo.png)

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

A preview of the [dataset](https://github.com/JayThibs/hyperdrive-vs-automl-plus-deployment/blob/main/Pump-it-Up-dataset.csv):

![Pump-it-Up-dataset.png](./imgs/Pump-it-Up-dataset.png)

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
