import mlflow


def run_main():
    parameters = {"alpha": 0.3}
    submitted_run = mlflow.projects.run(uri="./", entry_point="main", parameters=parameters)
    run_id = submitted_run.run_id
    mlflow_service = mlflow.tracking.MlflowClient()
    # run_infos = mlflow_service.list_run_infos(
    #     experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID,
    #     run_view_type=ViewType.ACTIVE_ONLY)
    run = mlflow_service.get_run(run_id)
    print("[run_main] run_id ", run_id, run.data.params)
    # assert run.data.params == parameters


def run_train_1():
    parameters = {"alpha": 0.4}
    submitted_run = mlflow.projects.run(uri="./", entry_point="train_1", parameters=parameters)
    run_id = submitted_run.run_id
    mlflow_service = mlflow.tracking.MlflowClient()
    # run_infos = mlflow_service.list_run_infos(
    #     experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID,
    #     run_view_type=ViewType.ACTIVE_ONLY)
    run = mlflow_service.get_run(run_id)
    print("[run_train_1] run_id ", run_id, run.data.params)
    # assert run.data.params == parameters


if __name__ == '__main__':
    run_main()
    run_train_1()