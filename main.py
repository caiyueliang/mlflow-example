import mlflow
import threading
from mlflow.tracking import MlflowClient
import mlflow.utils.mlflow_tags as mlflow_tags

from mlflow.entities import ViewType
import mlflow.tracking


class MlflowManager(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        pass

    # 设计成单例模式
    def __new__(cls, *args, **kwargs):
        print('[MlflowManager] __new__ 1')
        if not hasattr(MlflowManager, "_instance"):
            print('[MlflowManager] __new__ 2')
            with MlflowManager._instance_lock:
                print('[MlflowManager] __new__ 3')
                if not hasattr(MlflowManager, "_instance"):
                    print('[MlflowManager] __new__ 4')
                    MlflowManager._instance = object.__new__(cls)
                    MlflowManager._instance.__init()                # 初始化
        return MlflowManager._instance

    # 私有初始化函数
    def __init(self):
        tracking_uri = "file:./mlruns"
        artifact_location = None

        print('[MlflowManager] init start: tracking_uri: %s' % tracking_uri)
        self.artifact_location = artifact_location
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(uri=self.tracking_uri)
        print('[MlflowManager] init end ...')

    # =========================================================================
    def log_param(self, run_id, key, value):
        self.client.log_param(run_id, key, value)

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

    # =======================================================
    # 获取某个实验的某个主版本的所有子版本
    def get_sub_versions(self, experiment_id, version_id, view='active_only'):
        view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
        runs = self.client.store.search_runs([experiment_id], None, view_type)
        sub_versions_list = []
        for run in runs:
            tags = {k: v for k, v in run.data.tags.items()}
            parent_run_id = tags.get(mlflow_tags.MLFLOW_PARENT_RUN_ID, "")
            if parent_run_id == version_id:
                sub_versions_list.append(run)
        return sub_versions_list

    # 获取某个实验的某个主版本的子版本个数
    def get_sub_versions_count(self, experiment_id, version_id, view='active_only'):
        sub_versions_list = self.get_sub_versions(experiment_id, version_id, view)
        return len(sub_versions_list)

    # 判断子版本是不是主版本的子版本（孩子）
    def is_sub_versions(self, experiment_id, version_id, sub_version_name, view='active_only'):
        sub_versions_list = self.get_sub_versions(experiment_id, version_id, view)
        for sub_version in sub_versions_list:
            tags = {k: v for k, v in sub_version.data.tags.items()}
            run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")
            if run_name == sub_version_name:
                return True
        return False

    # 获取某个实验的某个主版本的某个子版本ID
    def get_sub_version_id(self, experiment_id, version_id, sub_version_name, view='active_only'):
        sub_versions_list = self.get_sub_versions(experiment_id, version_id, view)
        for sub_version in sub_versions_list:
            tags = {k: v for k, v in sub_version.data.tags.items()}
            run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")
            if run_name == sub_version_name:
                return sub_version.info.run_id
        return None

    # 自动生成一个子版本名称，格式如：V1.1, V1.2, V2.3
    def generate_sub_version_name(self, experiment_id, version_id, version_name):
        count = self.get_sub_versions_count(experiment_id=experiment_id, version_id=version_id)
        sub_version_name = version_name + "." + str(count + 1)
        return sub_version_name

    # =======================================================
    # 获取某个实验中，所有主版本
    def get_major_versions(self, experiment_id, view='active_only'):
        view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
        runs = self.client.store.search_runs([experiment_id], None, view_type)
        major_versions_list = []
        for run in runs:
            tags = {k: v for k, v in run.data.tags.items()}
            parent_run_id = tags.get(mlflow_tags.MLFLOW_PARENT_RUN_ID, "")
            if parent_run_id == "":
                print(tags.get(mlflow_tags.MLFLOW_RUN_NAME, ""))
                major_versions_list.append(run)
        return major_versions_list

    # 获取某个实验中，主版本个数
    def get_major_versions_count(self, experiment_id, view='active_only'):
        major_versions_list = self.get_major_versions(experiment_id, view)
        return len(major_versions_list)

    # 通过experiment_id和version_name查找主版本ID
    # view可选值: 'active_only', 'deleted_only', 'all'
    def get_major_version_id(self, experiment_id, version_name, view='active_only'):
        major_versions_list = self.get_major_versions(experiment_id, view)
        for major_version in major_versions_list:
            tags = {k: v for k, v in major_version.data.tags.items()}
            run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")
            if run_name == version_name:
                return major_version.info.run_id
        return None

    # 自动生成一个主版本名称，格式如：V1, V2, V3
    def generate_major_version_name(self, experiment_id):
        count = self.get_major_versions_count(experiment_id=experiment_id)
        major_version_name = "V" + str(count + 1)
        return major_version_name

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

    def get_run(self, run_id='27f2872ffe3144b59200350a83ac11a5'):
        run = self.client.get_run(run_id)
        print(run)
        print(run.data.params)


# def run_main(alpha=0.5):
#     parameters = {"alpha": alpha}
#     submitted_run = mlflow.projects.run(uri="./", entry_point="main", parameters=parameters)
#     run_id = submitted_run.run_id
#     mlflow_service = mlflow.tracking.MlflowClient()
#     # run_infos = mlflow_service.list_run_infos(
#     #     experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID,
#     #     run_view_type=ViewType.ACTIVE_ONLY)
#     run = mlflow_service.get_run(run_id)
#     print("[run_main] run_id ", run_id, run.data.params)
#
#
# def run_train_1(alpha=0.5):
#     parameters = {"alpha": alpha}
#     submitted_run = mlflow.projects.run(uri="./", entry_point="train_1", parameters=parameters)
#     run_id = submitted_run.run_id
#     mlflow_service = mlflow.tracking.MlflowClient()
#     # run_infos = mlflow_service.list_run_infos(
#     #     experiment_id=file_store.FileStore.DEFAULT_EXPERIMENT_ID,
#     #     run_view_type=ViewType.ACTIVE_ONLY)
#     run = mlflow_service.get_run(run_id)
#     print("[run_train_1] run_id ", run_id, run.data.params)


# def run(uri, entry_point="main", version=None, parameters=None,
#         experiment_name=None, experiment_id=None,
#         backend=None, backend_config=None, use_conda=True,
#         storage_dir=None, synchronous=True, run_id=None)


if __name__ == '__main__':
    # mlflow_manager = MlflowManager()
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

    count = MlflowManager().get_sub_versions_count(experiment_id='3', version_id='959f8a720e4e46e4946b718e40b990df')
    print(count)
    count = MlflowManager().get_sub_versions_count(experiment_id='3', version_id='c0f14bfc60e74f1c928e57f1d1d03aa7')
    print(count)
    count = MlflowManager().get_major_versions_count(experiment_id='3')
    print(count)
    count = MlflowManager().get_major_versions_count(experiment_id='4')
    print(count)
