import mlflow

mlflow.projects.run(uri="./", entry_point="main", parameters={"alpha": 0.3})
mlflow.projects.run(uri="./", entry_point="train_1", parameters={"alpha": 0.3})