import mlflow
from mlflow.tracking import MlflowClient
import mlflow.utils.mlflow_tags as mlflow_tags

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
            run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")

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
            run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")

            if run_name == version_name:
                run_id = run.info.run_id
                break

        return run_id

    # 获取某个实验中的某个版本的孩子个数
    def get_child_count(self, experiment_id, version_id, view='active_only'):
        count = 0
        view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
        runs = self.client.store.search_runs([experiment_id], None, view_type)
        for run in runs:
            tags = {k: v for k, v in run.data.tags.items()}
            parent_run_id = tags.get(mlflow_tags.MLFLOW_PARENT_RUN_ID, "")
            if parent_run_id == version_id:
                count += 1
        return count

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

    # 初始化版本，返回experiment_id和version_id
    # 函数逻辑：如果实验不存在，则返回None；如果版本不存在，则新建并返回id；如果版本存在，返回id
    def init_version(self, experiment_name, version_name):
        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment:
            version_id = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name)
            if not version_id:
                version_id = mlflow.start_run(experiment_id=experiment.experiment_id, run_name=version_name)
                print('[init_version] create version: %s in experiment: %s; ' % (experiment_name, version_name))
                return experiment.experiment_id, version_id.info.run_id
            else:
                print('[init_version] version: %s is exist in experiment: %s' % (version_name, experiment_name))
                return experiment.experiment_id, version_id
        else:
            print('[init_version] the experiment does not exist: %s' % experiment_name)
            return None, None

    # 创建一个一级版本
    def create_version_level_1(self, experiment_name, version_name):
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

    # 创建一个二级版本
    def create_version_level_2(self, experiment_name, version_name_1, version_name_2):
        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment:
            if self.is_version_name_exist(experiment_id=experiment.experiment_id, version_name=version_name_1):
                run_id = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name_1)
                if run_id:
                    with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id):
                        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=version_name_2, nested=True):
                            print('[success] experiment: %s; version: %s' % (experiment_name, version_name_2))
        else:
            print('[create_version][error] experiment_name: %s is not exist' % experiment_name)
            return False

    # 运行一个开发版本
    def run_develop_version(self, experiment_name, version_name, adjust=False, p1=0.1, p2=0.1):
        experiment = self.client.get_experiment_by_name(experiment_name)

        # 执行对应的任务
        if experiment:
            run_id = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name)
            if run_id:
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id):
                    if adjust is True:
                        parameters = {
                            'adjust_1': str(p1 * 2),
                            'adjust_2': str(p2 * 2),
                            'train_1': str(p1),
                            'train_2': str(p2),
                        }
                    else:
                        parameters = {
                            'train_1': str(p1),
                            'train_2': str(p2),
                        }
                    mlflow.log_params(parameters)
                    print('[run_develop_version][success]')
            else:
                print('[run_develop_version][error] version_name not found: %s' % version_name)
        else:
            print('[run_develop_version][error] experiment_name not found: %s' % experiment_name)

    # 运行一个离线|定时版本
    def run_offline_version(self, experiment_name, version_name, run_name, adjust=False, p1=0.1, p2=0.1):
        experiment = self.client.get_experiment_by_name(experiment_name)

        # 执行对应的任务
        if experiment:
            run_id = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name)
            if run_id:
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id, nested=True):
                    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, nested=True):
                        if adjust is True:
                            parameters = {
                                'adjust_1': str(p1 * 2),
                                'adjust_2': str(p2 * 2),
                                'train_1': str(p1),
                                'train_2': str(p2),
                            }
                        else:
                            parameters = {
                                'train_1': str(p1),
                                'train_2': str(p2),
                            }
                        mlflow.set_tag("tag", "cyl_tag")
                        mlflow.log_params(parameters)
            else:
                print('[run_offline_version][error] version_name not found: %s' % version_name)
        else:
            print('[run_offline_version][error] experiment_name not found: %s' % experiment_name)

    def run_offline_version_2(self, experiment_name, version_name_1, version_name_2, run_name, adjust=False, p1=0.1, p2=0.1):
        experiment = self.client.get_experiment_by_name(experiment_name)

        # 执行对应的任务
        if experiment:
            run_id_1 = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name_1)
            if run_id_1:
                with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id_1, nested=False):
                    run_id_2 = self.get_run_id(experiment_id=experiment.experiment_id, version_name=version_name_2)
                    if run_id_2:
                        with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_id_2, nested=True):
                            with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name, nested=True):
                                if adjust is True:
                                    parameters = {
                                        'adjust_1': str(p1 * 2),
                                        'adjust_2': str(p2 * 2),
                                        'train_1': str(p1),
                                        'train_2': str(p2),
                                    }
                                else:
                                    parameters = {
                                        'train_1': str(p1),
                                        'train_2': str(p2),
                                    }
                                mlflow.set_tag("tag", "cyl_tag")
                                mlflow.log_params(parameters)
                    else:
                        print('[run_offline_version][error] version_name not found: %s' % version_name_2)
            else:
                print('[run_offline_version][error] version_name not found: %s' % version_name_1)
        else:
            print('[run_offline_version][error] experiment_name not found: %s' % experiment_name)

    def get_run(self, run_id='27f2872ffe3144b59200350a83ac11a5'):
        run = self.client.get_run(run_id)
        print(run)
        print(run.data.params)


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
    # # =======================================================
    # # 创建一个实验
    # mlflow_manager.create_experiment(experiment_name='b_score')
    # mlflow_manager.create_experiment(experiment_name='qd_score')
    #
    # # =======================================================
    # # 创建一个版本
    # mlflow_manager.create_version(experiment_name='b_score', version_name='v1.0.0')
    # mlflow_manager.create_version(experiment_name='b_score', version_name='v2.0.0')
    # mlflow_manager.create_version(experiment_name='b_score', version_name='v3.0.0')
    #
    # # =======================================================
    # mlflow_manager.create_version(experiment_name='qd_score', version_name='v1.0.0_train')
    # mlflow_manager.create_version(experiment_name='qd_score', version_name='v1.0.0_adjust')
    # mlflow_manager.create_version(experiment_name='qd_score', version_name='v2.0.0_train')
    # mlflow_manager.create_version(experiment_name='qd_score', version_name='v2.0.0_adjust')

    # # =======================================================
    # # 旧：开发版本在一级更新（刷新），定时任务二级更新（新增）, 自动调參和模型训练在一条记录里（自动调參不是每次都执行）
    # mlflow_manager.run_develop_version(experiment_name='b_score', version_name='v1.0.0', adjust=True, p1=0.1, p2=0.1)
    # mlflow_manager.run_develop_version(experiment_name='b_score', version_name='v1.0.0', adjust=False, p1=0.9, p2=0.9)
    #
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v1.0.0', run_name='offline_1',
    #                                    adjust=True, p1=0.1, p2=0.1)
    #
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v1.0.0', run_name='offline_2',
    #                                    adjust=False, p1=0.2, p2=0.8)
    #
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v1.0.0', run_name='offline_3',
    #                                    adjust=True, p1=0.5, p2=0.5)

    # # =======================================================
    # # 新：开发版本在二级更新（新增），定时任务二级更新（新增）, 自动调參和模型训练在一条记录里（自动调參不是每次都执行）
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v3.0.0', run_name='v3.0.1',
    #                                    adjust=True, p1=0.1, p2=0.1)
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v3.0.0', run_name='v3.0.2',
    #                                    adjust=False, p1=0.2, p2=0.2)
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v3.0.0', run_name='offline_1',
    #                                    adjust=True, p1=0.9, p2=0.9)
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v3.0.0', run_name='offline_2',
    #                                    adjust=False, p1=0.4, p2=0.1)
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v3.0.0', run_name='v3.0.3',
    #                                    adjust=False, p1=0.2, p2=0.2)
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v3.0.0', run_name='offline_3',
    #                                    adjust=True, p1=0.4, p2=0.1)
    # mlflow_manager.run_offline_version(experiment_name='b_score', version_name='v3.0.0', run_name='v3.0.4',
    #                                    adjust=True, p1=0.2, p2=0.2)

    # # =======================================================
    # # 新：开发版本在二级更新（新增），定时任务二级更新（新增）, 自动调參和模型训练在一条记录里（自动调參不是每次都执行）
    # mlflow_manager.create_version_level_2(experiment_name='b_score', version_name_1='v2.0.0', version_name_2='v2.1.0')
    # mlflow_manager.create_version_level_2(experiment_name='b_score', version_name_1='v2.0.0', version_name_2='v2.2.0')
    # mlflow_manager.create_version_level_2(experiment_name='b_score', version_name_1='v2.0.0', version_name_2='offline')

    # mlflow_manager.run_offline_version_2(experiment_name='b_score', version_name_1='v2.0.0', version_name_2='v2.1.0',
    #                                      run_name='v2.1.1', adjust=True, p1=0.1, p2=0.1)

    # =======================================================
    # 运行一个离线|定时版本
    # run_offline_version(experiment_name='CYL_1', version_name='version_1', run_name='offile_1',
    #                     step_id=0, l=0.1, alpha=0.1)
    # run_offline_version(experiment_name='CYL_1', version_name='version_1', run_name='offile_2',
    #                     step_id=0, l=0.2, alpha=0.2)

    # run_offline_version(experiment_name='CYL_1', version_name='version_3', run_name='offile_1',
    #                     step_id=0, l=0.1, alpha=0.1)

    # =======================================================
    # mlflow_manager.get_run()

    # =======================================================
    # experiment_id, run_id = mlflow_manager.init_version(experiment_name='b_score', version_name='v3.0.0')
    # print(experiment_id, run_id)
    # run_id = mlflow_manager.init_version(experiment_name='b_score', version_name='v4.0.0')
    # print(run_id)
    # run_id = mlflow_manager.init_version(experiment_name='a_score', version_name='v3.0.0')
    # print(run_id)

    count = mlflow_manager.get_child_count(experiment_id='3', version_id='959f8a720e4e46e4946b718e40b990df')
    print(count)
    count = mlflow_manager.get_child_count(experiment_id='3', version_id='c0f14bfc60e74f1c928e57f1d1d03aa7')
    print(count)
