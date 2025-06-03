## Authors
M. Elasmi, Karlsruhe Institute of Technology\
F. Krumbiegel, Karlsruhe Institute of Technology

## Packages
1. Use Python3.11 in a virtual environment with, e.g., `python3.11 -m venv .venv`  
2. Activate the environment `source .venv/bin/activate`  
2.1. Upgrade pip `pip3 install --upgrade pip`  
3. Install the packages via `pip3 install -r requirements.txt`  

## Usage
1. In the folder examples, create a new example that should contain train_model.py and solve.py files. A new example requires a PDEcoefficient class. Either choose an existing one (see PDEcoefficient.py) or implement there a new one. Follow the structure of the provided examples or adapt the code accordingly
2. The configuration should be included in the train_model.py file
3. The current path should be examples/myExample
4. Start training using python3 train_model.py
5. To solve the pde, choose a trained case, define the corresponding loadPath and start python3 solve.py

## Visualize online progress with tensorboard
In a separate terminal, go to the example directory (/examples/myExample) and use the command
```
tensorboard --logdir results/myCase
```
then click on the link provided to see the graphs

## Information
This code is a proof of concept that conducts the examples in the paper "Neural Numerical Homogenization Based on Deep Ritz Corrections" by Mehdi Elasmi, Felix Krumbiegel, and Roland Maier. Note that this code is not optimized in any way.  
