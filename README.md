# ACSE-9: Hybrid Renewable Energy Power Control with Hybrid Storage

This is the directory for ACSE-2020 Final Year Project by Son Hoang Thanh.

## Project Description
This project implements reinforcement learning agent for power control of a choosen Hybrid Renewable Energy System (HRES) with hybrid storage. The model uses Deep Q-Learning method, with double two neural networks to find the optimal control policy. 

## Dependency Installation
The project was build and tested on Linux OS platform. The code contained in this repository is written in Python programming language. You can install the whole python development environment using Anaconda suite [here](https://www.anaconda.com/products/individual). The individual dependencies for this project can be installed using the requirements.txt file using pip3 Python package manager or conda package manager for Anaconda. For example:

`pip3 install -r requirements.txt `

## Repository structure
The repository is structured in the form of python package to allow for better clarity and modular form of each collection of functions. This makes it easier to import all the function across the whole scope of this repository.

# Data
The sources of data for this project are placed in the *./data* directory. The source of solar and wind power generation data for Isle of Man is [here](https://www.renewables.ninja/) with data from year 2019. The source of UK demand data after which the load data for Isle of Man is model is available [here](https://www.gridwatch.templar.co.uk/) for year 2019 and 2020. The powerplant sizing for this project were made with inference from HOMER software, with simulation files placed in *./HOMER* directory.

## Execution

Before running model training, hyperarameter tunning or model application, please go into the approriately named python script and edit the hyperparameter list in the main() function.


To run the model training on local CPU, type:

`python3 training.py`

To run hyperparameter tunning, run:

`python3 tuning.py`

To run pretrained model application, run:

`python3 application.py`

Please make sure that the pretrained model is appropriately named and can be found using the pth provided in model_path variable.;
## Testing

The components of the model can be tested using pytest unit testing placed in the *./tests*. To run the test, run:

`pytest`

or to run individual test files, run:

`pytest tests/[test_filename]`

## Results

The pretrained model state dictonary under name *trained_model_revenue* can be seen in *./results*.  The neural network manual prunning can also be viewed under *network_prunning.xlsx* file. The prunning was carried out from initially layered network in steps of 18 neurons. The hyperparameter search optimization were outputed into the *tune_results.csv*.

## Useful website

Isle of Man electricity price: https://www.manxutilities.im/your-home/electricity/domestic-tariffs/
Train Mario Example with Pytorch: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
Google Colab Q-learning Cheet Sheet: https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N#scrollTo=atoad6YFcKNV
Example OpenAI training environment for Cart Pole balancing problem: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py


