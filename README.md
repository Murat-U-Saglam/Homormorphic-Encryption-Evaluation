# Evaluating Homomorphic Encryption and its practical effects on implementation

### Requirements

All of the required packages for this program is within **requirements.txt**, this can be installed via

```
pip -r requirements.txt
```

The codebase can only be run on a linux environment

It is also heavily recomemmended the packages are installed within a virtual environment or it will clash with your core python packages.

The logical nature of the program is to generate the dataset with the **dataset_generator.py** file and then run the *ipynb files within the  **Data/EncryptedEvaluation** and **Data/EncryptedTraining.** However there is already a ready dataset.

The structure of the project is as follows

**LinearRegression**

Contains the folder with the models

**EncryptedEvaluation**

Contains the folder with the Encrypted Evaluation models along with it's respective data and unique requirements

**EncryptedTraining**

Contains the folder with the Encrypted Training models along with it's respective data and unique requirements

**Data**

Contains the output of **dataset_generator.py**

***_Log**

Contains the logs for the Encrypted Evaluation = EE and Encrypted Training = ET
