import mlflow
import itertools
from random import random as rand

def run_test():
    for l1, alpha in itertools.product([0.75, 1], [0, 0.5]):
        # with mlflow.start_run(run_id='91878d6666994fa8b7205f4a83171e8a', experiment_id=2, run_name='ipython'):
        with mlflow.start_run(run_id='91878d6666994fa8b7205f4a83171e8a', experiment_id=2, run_name='ipython'):
            parameters = {
                'l1': str(l1),
                'alpha': str(alpha),
            }
            # metrics = {
            #     'MAE': [rand()],
            #     'R2': [rand()],
            #     'RMSE': [rand()],
            # }
            mlflow.log_params(parameters)
            # mlflow.log_metrics(metrics)


def run_test_1(l1, alpha):
    with mlflow.start_run(run_id='00e2f575f9d84ad3b7f27ed106eecc80', experiment_id=2):
        with mlflow.start_run(run_name='Test', experiment_id=2, nested=True):
            parameters = {
                'l1': str(l1),
                'alpha': str(alpha),
            }
            # metrics = {
            #     'MAE': [rand()],
            #     'R2': [rand()],
            #     'RMSE': [rand()],
            # }
            mlflow.log_params(parameters)
            # mlflow.log_metrics(metrics)


def run_main(alpha=0.5):
    parameters = {"alpha": alpha}
    submitted_run = mlflow.projects.run(uri="./", entry_point="main", parameters=parameters)
    run_id = submitted_run.run_id
    mlflow_service = mlflow.tracking.MlflowClient()
    # run_infos = mlflow_service.list_run_infos(
    #     experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID,
    #     run_view_type=ViewType.ACTIVE_ONLY)
    run = mlflow_service.get_run(run_id)
    print("[run_main] run_id ", run_id, run.data.params)
    # assert run.data.params == parameters


def run_train_1(alpha=0.5):
    parameters = {"alpha": alpha}
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
    # run_test()
    run_test_1(0.1, 0.1)
    # run_main(alpha=0.4)
    # run_train_1()
