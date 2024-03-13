# The `age` module

The age module is the machine learning part of the project which cointains functions and classes for bone age estimations. If you decide to use just this module for example if you want to implement your custom model or train our model in your custom dataset, here you can explaination of what is in age module ad how it works. Age module structure is the following one:

```bash
-- boneagessessment
    | -- baa
        | -- RSNA
        | -- preprocessing
-->     | -- age 
            | -- __init__.py
            | -- age_macro.json
            | -- age.py
            | -- model.py
            | -- results
                | -- case_a
                | -- case_b
                | -- case_c
                | -- case_d
                | -- case_e
                | -- case_f
                | -- best_case
                | -- example
            | -- unittest
                | -- age_test.py
            | -- weights
                | -- best_model.keras
```
- `__init__.py`

This empty function must be into age module to recall in other part of code.
- `age_macro.json`

This json file cointains variables that could be change during training. This file is the input of `age.py`. 

- `age.py`

This function is the main of the module. After you moved to age directory with `cd`, run this python code you have three options:
1. The first one is just simple run the script. If you choose this option, default parameters for training are used; where defaul parameters are:
```python
params = {
    'Learning rate': 1e-05,
    'Regularization factor': 1e-04,
    'Batch size': (32, 32, 1396),
    'Number of epochs': 20
    }
```
```bash
python3 age.py
```
2. The second option is to write by keyboard the parameters using the flag `--keyboard_input` and follow the instrunctions.
```bash
python3 age.py --keyboard_input
```
3. The last and **reccomended** one is to update the json file and use the flag `--macro` followed by json file name.
```bash
python3 age.py --macro age_macro.json
```

- `model.py`

This file is the real heart of the machine learning module. It cointains statistics function for model evaluation like `mean_absolute_error` and `mean_absolute_deviation`, but also two importante classes. The first one is `BaaModel` where you can find as methods the different versions of our rVGG16 model. The second one is `BoneAgeAssessment` which cointains several methods to speed up develpment and make a more user-friendly interface. For example, thanks to this class, you can easily train the model with few lines of code:
```python
from age import BoneAgeAssessment
from age import BaaModel as Model

# Create a BoneAgeAssessment instance and updates values
baa = BoneAgeAssessment()
baa.__update_batch_size__(BATCH_SIZE)
baa.__update_epochs__(EPOCHS)
baa.__update_lr__(LEARNING_RATE)
baa. __change_training__(balanced=True)

# Show info
baa.__show_info__()

# Now create the model
model = Model.vgg16regression_l2(REG_FACTOR)

# Compile the model
baa.compiler(model)

# Start training
baa.training_evaluation(model)

# Test the model with best weights of last model
WEIGHTS_NAME = '../best_model.keras'
PATH_TO_WEIGHTS = os.path.join(WEIGHTS_NAME)
baa.model_evaluation(PATH_TO_WEIGHTS)
```
- weights

This directory contains `best_model.keras`.  

- results 

In results folder you can find results of our training changing hyperameters. For more information about choosen parameters read the report.

- unittest

Test folder for simple unittests in the `age_test.py` script.
