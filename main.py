import mlflow
import threading
from mlflow.tracking import MlflowClient
import mlflow.utils.mlflow_tags as mlflow_tags

from mlflow.entities import ViewType
import mlflow.tracking
import json
import traceback


class MlflowManager(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if self._init_flag is False:
            tracking_uri = "file:./mlruns"
            # tracking_uri = "mysql://test:test@10.117.61.106:3306/qdmlflow?charset=utf8"
            artifact_location = None

            print('[MlflowManager] init start: tracking_uri: %s' % tracking_uri)
            self.artifact_location = artifact_location
            self.tracking_uri = tracking_uri
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            mlflow.set_tracking_uri(uri=self.tracking_uri)
            print('[MlflowManager] init end ...')
            self._init_flag = True
        return

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
        return MlflowManager._instance

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
    # 获取某个实验的某个主版本的所有子版本（孩子）
    def get_minor_versions(self, experiment_id, version_id, view='active_only'):
            view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
            runs = self.client.store.search_runs([experiment_id], None, view_type)
            minor_versions_list = []
            for run in runs:
                tags = {k: v for k, v in run.data.tags.items()}
                parent_run_id = tags.get(mlflow_tags.MLFLOW_PARENT_RUN_ID, "")
                if parent_run_id == version_id:
                    minor_versions_list.append(run)
            return minor_versions_list

    # 获取某个实验的某个主版本的子版本（孩子）个数
    def get_minor_versions_count(self, experiment_id, version_id, view='active_only'):
            minor_versions_list = self.get_minor_versions(experiment_id, version_id, view)
            return len(minor_versions_list)

    # 判断子版本是不是主版本的子版本（孩子）
    def is_minor_version(self, experiment_id, version_id, minor_version_name, view='active_only'):
            minor_versions_list = self.get_minor_versions(experiment_id, version_id, view)
            for minor_version in minor_versions_list:
                tags = {k: v for k, v in minor_version.data.tags.items()}
                run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")
                if run_name == minor_version_name:
                    return True
            return False

    # 获取某个实验的某个主版本的某个子版本ID
    def get_minor_version_id(self, experiment_id, version_id, minor_version_name, view='active_only'):
            minor_versions_list = self.get_minor_versions(experiment_id, version_id, view)
            for minor_version in minor_versions_list:
                tags = {k: v for k, v in minor_version.data.tags.items()}
                run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")
                if run_name == minor_version_name:
                    return minor_version.info.run_id
            return None

    # 自动生成一个子版本名称，格式如：1.1, 1.2, 2.3
    def generate_minor_version_name(self, experiment_id, version_id, version_name):
            count = self.get_minor_versions_count(experiment_id=experiment_id, version_id=version_id)
            minor_version_name = version_name + "." + str(count + 1)
            return minor_version_name

    # 清理多余的子版本（按时间清理，保留运行时间最近的n个）
    def clean_minor_versions(self, experiment_id, version_id, reserve_count=4):
        try:
            minor_versions = self.get_minor_versions(experiment_id=experiment_id, version_id=version_id)
            if len(minor_versions) <= reserve_count:
                return

            # minor_versions按时间有序排列，只需要遍历后面的几个，直接删除即可
            for i in range(reserve_count, len(minor_versions)):
                print('[minor_version]', i, minor_versions[i].info.experiment_id, minor_versions[i].info.run_id,
                      minor_versions[i].info.start_time)
                self.client.delete_run(run_id=minor_versions[i].info.run_id)
        except Exception as e:
            content = '[clean_minor_versions][error] {}'.format(e)
            content = '[{}] {} {}'.format(experiment_id, version_id, content)
            print(content)

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

    # 自动生成一个主版本名称，格式如：1, 2, 3
    def generate_major_version_name(self, experiment_id):
            count = self.get_major_versions_count(experiment_id=experiment_id)
            major_version_name = str(count + 1)
            return major_version_name

    # =======================================================
    def get_versions(self, experiment_id, view='active_only'):
        view_type = ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY
        runs = self.client.store.search_runs([experiment_id], None, view_type)
        return runs

    # 通过experiment_id和version_name查找版本ID，不区分主版本或子版本
    def get_version_id(self, experiment_id, version_name, view='active_only'):
        runs = self.get_versions(experiment_id, view)
        for run in runs:
            tags = {k: v for k, v in run.data.tags.items()}
            run_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")
            if run_name == version_name:
                return run.info.run_id
        return None

    # 获取最优的版本
    def get_best_version(self, experiment_name, metrics='rmse'):
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            best_version_name = None
            best_version_id = None
            best_value = 0.0

            version_list = self.get_versions(experiment_id)
            print(len(version_list))
            for version in version_list:
                if metrics in version.data.metrics.keys():
                    print(version.data.metrics[metrics])
                    if version.data.metrics[metrics] > best_value:
                        best_value = version.data.metrics[metrics]
                        tags = {k: v for k, v in version.data.tags.items()}
                        best_version_name = tags.get(mlflow_tags.MLFLOW_RUN_NAME, "")
                        best_version_id = version.info.run_id
            return best_version_name, best_version_id
        else:
            print('[get_best_version][error] experiment not found: %s' % experiment_name)
            return None, None

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

    # =================================================================
    # 运行一个新的主版本，新增一条运行记录
    def create_and_run_major_version(self, experiment_name, func, args):
            experiment = self.client.get_experiment_by_name(experiment_name)
            # 判断实验是否存在
            if experiment:
                experiment_id = experiment.experiment_id
                # 自动生成一个主版本名称
                major_version_name = self.generate_major_version_name(experiment_id)

                run = mlflow.start_run(experiment_id=experiment_id, run_name=major_version_name)
                if run:
                    print('[run_major_version_new] start experiment: %s major_version: %s' %
                          (experiment_name, major_version_name))
                    # 运行函数
                    func(experiment_name=experiment_name, experiment_id=experiment_id,
                         version_name=major_version_name, version_id=run.info.run_id, args=args)
                mlflow.end_run()
            else:
                print('[run_major_version_new][error] experiment not found: %s' % experiment_name)

    # 运行一个新的子版本，新增一条运行记录
    def create_and_run_minor_version(self, experiment_name, version_name, func, args):
        experiment = self.client.get_experiment_by_name(experiment_name)
        # 判断实验是否存在
        if experiment:
            experiment_id = experiment.experiment_id
            version_id = self.get_major_version_id(experiment_id=experiment_id, version_name=version_name)
            # 主版本是否存在
            if version_id:
                with mlflow.start_run(experiment_id=experiment_id, run_id=version_id):
                    # 自动生成一个子版本名称
                    minor_version_name = self.generate_minor_version_name(experiment_id, version_id, version_name)
                    run = mlflow.start_run(experiment_id=experiment_id, run_name=minor_version_name, nested=True)
                    if run:
                        print(
                            '[run_minor_version_new] start experiment: %s major_version: %s minor_version_name: %s' %
                            (experiment_name, version_name, minor_version_name))
                        # 运行函数
                        func(experiment_name=experiment_name, experiment_id=experiment_id,
                             version_name=minor_version_name, version_id=run.info.run_id, args=args)
                    mlflow.end_run()
            else:
                print('[run_minor_version_new][error] major_version not found: %s' % version_name)
        else:
            print('[run_minor_version_new][error] experiment not found: %s' % experiment_name)


def log_test(a=0.5):
    mlflow.log_param('alpha_1', 0.1)
    mlflow.log_param('alpha_2', 0.2)
    mlflow.log_metric('mse', a)


# ======================================================================================================================
# 递归替换json文本
def replace_json_text(base_json, replace_json):
    if isinstance(base_json, dict) and isinstance(replace_json, dict):
        for key in base_json:
            # print('[replace_json]', key, replace_json.keys())
            if key in replace_json.keys():
                if not isinstance(base_json[key], dict) and not isinstance(base_json[key], list):
                    if base_json[key] != replace_json[key]:
                        print("[replace_json] key:%s  value:%s new_value:%s" % (key, base_json[key], replace_json[key]))
                        base_json[key] = replace_json[key]
                replace_json_text(base_json[key], replace_json[key])
    elif isinstance(base_json, list) and isinstance(replace_json, list):
        for base_item in base_json:
            for replace_item in replace_json:
                print('[replace_json] ', replace_item)
                replace_json_text(base_item, replace_item)


# 替换配置文件中的某些字段
def replace_file(params_file, replace_params_file):
    try:
        print('[replace_file] %s' % params_file)
        print('[replace_file] %s' % replace_params_file)
        with open(params_file, 'r') as f:
            base_params = f.read()
            base_params = json.loads(base_params)

        with open(replace_params_file, 'r') as f:
            replace_params = f.read()
            replace_params = json.loads(replace_params)

        print('[replace_file] base_params old %s' % base_params)
        print('[replace_file] replace_params %s' % replace_params)
        replace_json_text(base_params, replace_params)
        print('[replace_file] base_params new %s' % base_params)

        # 替换后的值重新写入 params_file
        # with open(params_file, 'w') as f:
        #     base_params = json.dumps(base_params)
        #     f.write(base_params)
    except Exception as e:
        traceback.print_exc()
        content = '[replace_file][error] {}'.format(e)
        print(content)
        exit(-1)

    return


if __name__ == '__main__':
    # replace_file(params_file='/Users/qudian/qudian-ml/qdmlflow/experiments_config/1/conf/model_params.conf',
    #              replace_params_file='/Users/qudian/qudian-ml/qdmlflow/bin/params_file.json')

    MlflowManager().clean_minor_versions(experiment_id='3', version_id='959f8a720e4e46e4946b718e40b990df')

    # name, id = MlflowManager().get_best_version(experiment_name='Test_1')
    # print(name, id)
    # # =======================================================
    # # 创建一个实验
    # MlflowManager().create_experiment(experiment_name='b_score_test')
    # MlflowManager().create_experiment(experiment_name='qd_score')
    # #
    # # # =======================================================
    # # # 创建一个版本
    # MlflowManager().create_and_run_major_version(experiment_name='b_score_test', func=log_test, args=0.4)
    # MlflowManager().create_and_run_major_version(experiment_name='b_score_test', func=log_test, args=0.8)
    # MlflowManager().create_and_run_major_version(experiment_name='b_score_test', func=log_test, args=0.9)
    # #
    # # # =======================================================
    # MlflowManager().create_and_run_minor_version(experiment_name='b_score_test', version_name="1", func=log_test,
    #                                              args=0.9)
    # MlflowManager().create_and_run_minor_version(experiment_name='b_score_test', version_name="1", func=log_test,
    #                                              args=0.8)
    #
    # MlflowManager().create_and_run_minor_version(experiment_name='b_score_test', version_name="2", func=log_test,
    #                                              args=0.5)
    #
    # MlflowManager().create_and_run_minor_version(experiment_name='b_score_test', version_name="2", func=log_test,
    #                                              args=0.4)
    # MlflowManager().create_and_run_minor_version(experiment_name='b_score_test', version_name="2", func=log_test,
    #                                              args=0.7)
    #
    # MlflowManager().create_and_run_minor_version(experiment_name='b_score_test', version_name="3", func=log_test,
    #                                              args=0.9)
    # =======================================================
    # experiment_id, run_id = mlflow_manager.init_version(experiment_name='b_score', version_name='v3.0.0')
    # print(experiment_id, run_id)
    # run_id = mlflow_manager.init_version(experiment_name='b_score', version_name='v4.0.0')
    # print(run_id)
    # run_id = mlflow_manager.init_version(experiment_name='a_score', version_name='v3.0.0')
    # print(run_id)

    # count = MlflowManager().get_sub_versions_count(experiment_id='3', version_id='959f8a720e4e46e4946b718e40b990df')
    # print(count)
    # count = MlflowManager().get_sub_versions_count(experiment_id='3', version_id='c0f14bfc60e74f1c928e57f1d1d03aa7')
    # print(count)
    # count = MlflowManager().get_major_versions_count(experiment_id='3')
    # print(count)
    # count = MlflowManager().get_major_versions_count(experiment_id='4')
    # print(count)
