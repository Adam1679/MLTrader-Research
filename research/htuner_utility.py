import sys
from curses.ascii import isalnum
from tkinter import Scale
from typing import Dict

from bytedhtuner.htuner_client import HTunerClient, StudyConfigFactory, TrialContainer
from bytedhtuner.idl.hyperparameter_tuner_thrift import (
    BayesianAlgorithmConfig,
    MetricConstraint,
    ScaleType,
)


def get_htuner_client(client_id=-1):
    return HTunerClient(
        client_id=client_id, model_output="./", host="localhost", port=5608
    )


def suggest_trial(study_id, worker_id):
    htuner_client = get_htuner_client(client_id=worker_id)
    htuner_client.link_study_by_study_id(study_id)
    res = htuner_client.suggest_trial(max_retries=10)
    if res["status"] == "SUGGESTED":
        trial_id = res["trial"].trial_id
        hparams = res["trial"].params
        return trial_id, hparams
    else:
        return None, None


def report_metric(
    study_id, trial_id, hparams, metrics: Dict[str, float] = None, infeasible=False
):
    htuner_client = get_htuner_client(client_id=0)
    htuner_client.link_study_by_study_id(study_id)
    if infeasible:
        htuner_client.complete_trial(trial_id=trial_id, metric=None, infeasible=True)
        print(f"[{study_id}.{trial_id}] INFEASIBLE! hparams={hparams}\n")
        return
    constraint_list_names = []
    if htuner_client.study_config.bayesian_algorithm_config.metric_constraints:
        for (
            metric
        ) in htuner_client.study_config.bayesian_algorithm_config.metric_constraints:
            constraint_list_names.append(metric.metric_name)

    constraint_metric = {}
    for name in constraint_list_names:
        constraint_metric[name] = metrics[name]

    obj = metrics["objective_value"]
    htuner_client.complete_trial(
        trial_id=trial_id,
        metric=obj,
        infeasible=False,
        constrained_metrics=constraint_metric or None,
    )
    print(
        f"[{study_id}.{trial_id}] objective_value={obj}, metrics={constraint_metric} hparams={hparams}\n"
    )
    return


def htuner_register_study(
    parameters, study_name, max_num_trials, parameter_constraints=None
):
    constraints = []
    constraints.append(MetricConstraint(metric_name="# Trades/Day", min_value=1))
    # constraints.append(MetricConstraint(metric_name="Win Rate [%]", min_value=0.4))
    constraints.append(MetricConstraint(metric_name="Profit Factor", min_value=1))
    # constraints.append(MetricConstraint(metric_name="Calmar Ratio", min_value=1.0))
    bayesian_algorithm_config = BayesianAlgorithmConfig(metric_constraints=constraints)
    study_config_factory = StudyConfigFactory(
        user_name="anxiang.zhang",
        study_name=study_name,
        max_num_trials=max_num_trials,
        bayesian_algorithm_config=bayesian_algorithm_config,
        parameter_constraints=parameter_constraints,
    )
    for param_spec in parameters:
        pname = param_spec["name"]
        if param_spec["type"] == "choice":
            if param_spec.get("is_ordered", False):
                study_config_factory.add_discrete_param(
                    pname,
                    param_spec["values"],
                    scale_type=ScaleType.UNIT_LOG_SCALE
                    if param_spec.get("log_scale", False)
                    else ScaleType.UNIT_LINEAR_SCALE,
                )

            else:
                study_config_factory.add_categorical_param(
                    pname,
                    param_spec["values"],
                    scale_type=ScaleType.UNIT_LOG_SCALE
                    if param_spec.get("log_scale", False)
                    else ScaleType.UNIT_LINEAR_SCALE,
                )
        elif param_spec["type"] == "range":
            study_config_factory.add_float_param(
                pname,
                min_value=param_spec["bounds"][0],
                max_value=param_spec["bounds"][1],
                scale_type=ScaleType.UNIT_LOG_SCALE
                if param_spec.get("log_scale", False)
                else ScaleType.UNIT_LINEAR_SCALE,
            )
        else:
            raise NotImplementedError()
    study_config = study_config_factory.get_study_config()
    htuner_client = get_htuner_client(client_id=0)
    rsp = htuner_client.create_study(study_config)
    study_id = rsp.study_id
    return study_id


def visulaize_results_to_neptune(study_id):
    htuner_client = get_htuner_client(client_id=0)
    htuner_client.link_study_by_study_id(study_id)
    trials = htuner_client.list_all_trials()
    print(f"study_id: {study_id}")
    for trial in trials:
        trial_container = TrialContainer(trial)
        print(f"trial id: {trial_container.trial_id}")
        neptune_client = neptune.init(
            project="adamzhang1679/MLTradeAutoML",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MjJkMzM1Mi0yYjdiLTRhMjMtODkwZC1iOTczYzU2YjJmYmEifQ==",
            tags=[str(study_id)],
            custom_run_id=f"{study_id}.{trial_container.trial_id}",
        )
        neptune_client["hparams"] = trial_container.params
        neptune_client["trial_id"] = str(trial_container.trial_id)
        if not trial.infeasible:
            metrics = {}
            metrics["objective_value"] = trial.metric_holder.objective_value
            for metric in trial.metric_holder.constrained_metrics:
                metrics[metric.name] = metric.value
            for key, value in metrics.items():
                value = float(str(value))
                if value > 99999:
                    metrics[key] = 99999
                elif value < -99999:
                    metrics[key] = -99999

            neptune_client["metrics"] = metrics
            neptune_client["infeasible"] = False
        else:
            neptune_client["infeasible"] = False
        neptune_client.stop()


if __name__ == "__main__":
    import neptune.new as neptune

    study_id = int(sys.argv[1])
    visulaize_results_to_neptune(study_id)
