import mlflow


if __name__ == '__main__':

    for l1, alpha in itertools.product([0, 0.25, 0.5, 0.75, 1], [0, 0.5, 1]):
        with mlflow.start_run(run_name='ipython'):
            parameters = {
                'l1': str(l1),
                'alpha': str(alpha),
            }
            metrics = {
                'MAE': [rand()],
                'R2': [rand()],
                'RMSE': [rand()],
            }
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)