import mlflow
from mlflow.models import infer_signature

"""
Creates a trained model to classif
"""
import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set our tracking server uri for logging
# start server if running locally - mlflow server --host 127.0.0.1 --port 8081
MLFLOW_TRACK_URI = "http://127.0.0.1:8081"
EXP_NAME = "MLflow Quickstart"

def if_directory_exists(dir1):
  
    if not os.path.isdir(dir1):
        raise Exception(f"dir {dir1} not found ")
   

class LogisticRegr:
   def __init__(self):
        pass
   
   def load_dataset(self):
       # Load the Iris dataset
       X, y = datasets.load_iris(return_X_y=True)
       #print(X)
       self.X = X
       self.y = y
       
   def create_training(self):
       # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

   def initialize_train_model(self):
       # TODO move to calling 
       # Define the model hyperparameters
        params = {
            "solver": "lbfgs",
            "max_iter": 1000,
            "multi_class": "auto",
            "random_state": 8888,
        }

        self.params = params

        # Train the model
        self.lr = LogisticRegression(**params)
        
        self.lr.fit(self.X_train, self.y_train)
       
   def get_accuracy_score(self):
       # Calculate metrics
       # Predict on the test set
       self.y_pred = self.lr.predict(self.X_test)
       self.accurracy = accuracy_score(self.y_test, self.y_pred)
       return self.accurracy

   def start_training(self):
      self.load_dataset()
      self.create_training()
      self.initialize_train_model()
      self.get_accuracy_score()
       
   # TODO create variables
   def mlflow_logging(self):
       
        # Set our tracking server uri for logging
        # start server if running locally - mlflow server --host 127.0.0.1 --port 8081
        #TODO create excpetion if failed
        #mlflow.set_tracking_uri(uri=MLFLOW_TRACK_URI)

        # Create a new MLflow Experiment
        mlflow_exp = mlflow.set_experiment(EXP_NAME)

        #self.mlflow_exp_id =  mlflow_exp.experiment_id
        #print(exp.experiment_id)

        # Start an MLflow run
        with mlflow.start_run():
            # Log the hyperparameters
            mlflow.log_params(self.params)

            # Log the loss metric
            mlflow.log_metric("accuracy", self.accurracy)
            

            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", "Basic LR model for iris data")

            # Infer the model signature
            signature = infer_signature(self.X_train, self.lr.predict(self.X_train))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=self.lr,
                # TODO make this a variable
                artifact_path="iris_model",
                signature=signature,
                input_example=self.X_train,
                # TODO make this a variable
                registered_model_name="tracking-quickstart",
            )
            # mod path example
            # ./mlartifacts/611172340721408599/0e939969c0d4474085b33536c823fac8/artifacts/iris_model
            """
            Some notes:
                # server model
                mlflow models serve -m ./mlartifacts/611172340721408599/0e939969c0d4474085b33536c823fac8/artifacts/iris_model -p 8082 --no-conda
                exp.experiment_id
            '611172340721408599'
             model_info.run_id
            '38f90eca3e9942259bd81264dfd3e472'
            model_info.artifact_path
            'iris_model'
            """
            #mod_path = "./mlartifacts/" + mlflow_exp.experiment_id + "/" + \
            #   model_info.run_id + "/" + "artifacts" + "/" + model_info.artifact_path
            mod_path = "./mlruns/" + mlflow_exp.experiment_id + "/" + \
               model_info.run_id + "/" + "artifacts" + "/" + model_info.artifact_path
            return mod_path



def main ():

    lr = LogisticRegr()
    lr.start_training()
    # model directory
    dir_model = lr.mlflow_logging()
    if_directory_exists(dir_model)
    print(dir_model)

main()

#TODO use logging method to print   


def test_dir_exists():
    if_directory_exists("blah")
def test_data_clean():
    """ Validates input data passed is correct"""
def test_model():

  lr = LogisticRegr()
  #lr.start_training()
  #print(lr.get_accuracy_score())
  #assert lr.get_accuracy_score() > 0.8
  #print(lr.mlflow_logging())








