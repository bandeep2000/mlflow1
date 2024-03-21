
FROM ghcr.io/mlflow/mlflow

# Run as docker build -t test-ml . --build-arg model_artifact=<model_artifact_path>
ARG model_artifact
ENV model_a $model_artifact


RUN apt update 
#RUN apt install curl

# Set environment variables
ENV MLFLOW_HOME=/mlflow \
    MLFLOW_SERVER_PORT=8082

# Install MLflow
#RUN pip install mlflow

# Expose MLflow server port
#EXPOSE $MLFLOW_SERVER_PORT

# Set the working directory for MLflow
WORKDIR $MLFLOW_HOME

CMD echo "My environment variable value is: $model_a"

#COPY ./mlartifacts/611172340721408599/0e939969c0d4474085b33536c823fac8/artifacts/iris_model .
COPY $model_a .

# Command to start MLflow server
CMD ["mlflow","models", "serve", "-m", ".", "-h", "0.0.0.0", "-p", "8082", "--no-conda"]
#mlflow models serve -m /Users/ban/ml1/mlflow/mlartifacts/611172340721408599/0e939969c0d4474085b33536c823fac8/artifacts/iris_model -p 8082 --no-conda

EXPOSE 8082



