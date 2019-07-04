import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from mlflow.entities import ViewType
import mlflow.tracking


class MlflowManager(object):
    def __init__(self):
        self.client = MlflowClient()

    # 判断实验是否存在
    def is_experiment_exist(self, experiment_name):
        find_experiment = self.client.get_experiment_by_name(experiment_name)
        if find_experiment:
            return True
        else:
            return False

    # 判断version_name是否存在
    def is_version_name_exist(self, experiment_id, version_name, view='active_only'):
        view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
        runs = self.client.store.search_runs([experiment_id], None, view_type)
        for run in runs:
            tags = {k: v for k, v in run.data.tags.items()}
            run_name = tags.get(MLFLOW_RUN_NAME, "")

            if run_name == version_name:
                return True

        return False

    # 通过experiment_id和version_name查找run_id
    def get_run_id(self, experiment_id, version_name, view='active_only'):    # ['active_only', 'deleted_only', 'all']
        run_id = None

        view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
        runs = self.client.store.search_runs([experiment_id], None, view_type)
        for run in runs:
            tags = {k: v for k, v in run.data.tags.items()}
            run_name = tags.get(MLFLOW_RUN_NAME, "")

            if run_name == version_name:
                run_id = run.info.run_id
                break

        return run_id

    # =======================================================
    # 创建一个实验
    def create_experiment(self, experiment_name):
        if not self.is_experiment_exist(experiment_name):
            experiment_id = self.client.create_experiment(name=experiment_name, artifact_location=None)
            print('[create_experiment][success] experiment_name: %s; experiment_id: %s' %
                  (experiment_name, experiment_id))
            return True
        else:
            print('[create_experiment][error] experiment_name: %s is exist' % experiment_name)
            return False

    # 创建一个版本
    def create_version(self, experiment_name, version_name):
        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment:
            if not self.is_version_name_exist(experiment_id=experiment.experiment_id, version_name=version_name):
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=version_name):
                    print('[create_version][success] experiment_name: %s; version_name: %s' %
                          (experiment_name, version_name))
                    return True
            else:
                print('[create_version][error] version_name: %s is exist in %s' % (version_name, experiment_name))
                return False
        else:
            print('[create_version][error] experiment_name: %s is not exist' % experiment_name)
            return False

    # 运行一个开发版本
    def run_develop_version(self, experiment_name, version_name, step_id=0, l=0.1, alpha=0.1):
        experiment = self.client.get_experiment_by_name(experiment_name)

        # 执行对应的任务
        if experiment:
            run_id = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name)
            if run_id:
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id):
                    if step_id == 0:
                        parameters = {
                            'l1': str(l),
                            'alpha1': str(alpha),
                        }
                    else:
                        parameters = {
                            'l2': str(l),
                            'alpha2': str(alpha),
                        }
                    mlflow.log_params(parameters)
                    print('[run_develop_version][success]')
            else:
                print('[run_develop_version][error] version_name not found: %s' % version_name)
        else:
            print('[run_develop_version][error] experiment_name not found: %s' % experiment_name)

    # 运行一个离线|定时版本
    def run_offline_version(self, experiment_name, version_name, run_name, step_id=0, l=0.1, alpha=0.1):
        experiment = self.client.get_experiment_by_name(experiment_name)

        # 执行对应的任务
        if experiment:
            run_id = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name)
            if run_id:
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id):
                    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, nested=True):
                        if step_id == 0:
                            parameters = {
                                'l1': str(l),
                                'alpha1': str(alpha),
                            }
                        else:
                            parameters = {
                                'l2': str(l),
                                'alpha2': str(alpha),
                            }
                        mlflow.set_tag("tag", "cyl_tag")
                        mlflow.log_params(parameters)
            else:
                print('[run_offline_version][error] version_name not found: %s' % version_name)
        else:
            print('[run_offline_version][error] experiment_name not found: %s' % experiment_name)


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


# def run(uri, entry_point="main", version=None, parameters=None,
#         experiment_name=None, experiment_id=None,
#         backend=None, backend_config=None, use_conda=True,
#         storage_dir=None, synchronous=True, run_id=None)


if __name__ == '__main__':
    mlflow_manager = MlflowManager()
    # =======================================================
    # 创建一个实验
    # mlflow_manager.create_experiment(experiment_name='CYL_1')
    mlflow_manager.create_experiment(experiment_name='CYL_2')

    # =======================================================
    # 创建一个版本
    mlflow_manager.create_version(experiment_name='CYL_2', version_name='version_1')
    # create_version(experiment_name='CYL_1', version_name='version_2')
    # create_version(experiment_name='CYL_1', version_name='version_3')

    # =======================================================
    # 运行一个开发版本
    # run_develop_version(experiment_name='CYL_1', version_name='version_1', step_id=0, l=0.1, alpha=0.1)
    # run_develop_version(experiment_name='CYL_1', version_name='version_1', step_id=1, l=0.9, alpha=0.9)

    # run_develop_version(experiment_name='CYL_1', version_name='version_2', step_id=0, l=0.3, alpha=0.3)
    # run_develop_version(experiment_name='CYL_1', version_name='version_2', step_id=1, l=0.9, alpha=0.9)

    # =======================================================
    # 运行一个离线|定时版本
    # run_offline_version(experiment_name='CYL_1', version_name='version_1', run_name='offile_1',
    #                     step_id=0, l=0.1, alpha=0.1)
    # run_offline_version(experiment_name='CYL_1', version_name='version_1', run_name='offile_2',
    #                     step_id=0, l=0.2, alpha=0.2)

    # run_offline_version(experiment_name='CYL_1', version_name='version_3', run_name='offile_1',
    #                     step_id=0, l=0.1, alpha=0.1)
