import mlflow

mlflow.projects.run(uri="./", entry_point="main", parameters={"alpha": 0.3})
