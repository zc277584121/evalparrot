from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from ragas.metrics.base import Metric

import time
from ragas import evaluate

from datasets import Dataset
from langchain.callbacks import get_openai_callback
from ragas.evaluation import Result


def retry_evaluate(
        dataset: Dataset,
        metrics: list[Metric] | None = None,
        column_map: dict[str, str] = {
            "question": "question",
            "contexts": "contexts",
            "answer": "answer",
            "ground_truths": "ground_truths",
        },
        retry_num=3,
        retry_interval=60 * 2
) -> Result:
    with get_openai_callback() as cb:
        for _ in range(retry_num):
            try:
                result = evaluate(
                    dataset,
                    metrics=metrics,
                    column_map=column_map
                )
                break
            except Exception as e:
                time.sleep(retry_interval)
                print(e)
                print('failed, retry...')
                continue
        print(f' this running token = {cb.total_tokens}, cost about = ${cb.total_cost}.')
    print(result)
    return result


def calcu_mean_var(result_list):
    name_2_scores = defaultdict(list)
    for result in result_list:
        for name, score in result.items():
            name_2_scores[name].append(score)
    multi_run_result = dict()
    for name, scores in name_2_scores.items():
        mean = np.mean(scores)
        var = np.var(scores)
        multi_run_result[name] = {'mean': mean, 'var': var}
    for ind, result in enumerate(result_list):
        result['ind'] = ind
    return multi_run_result


def multi_evaluate_one_dataset(
        dataset: Dataset,
        metrics: list[Metric] | None = None,
        column_map: dict[str, str] = {
            "question": "question",
            "contexts": "contexts",
            "answer": "answer",
            "ground_truths": "ground_truths",
        },
        run_num=4,
        retry_num=3,
        retry_interval=60 * 2,
) -> Tuple[List[Result], Dict]:
    print('run multi_evaluate_one_dataset()')
    t0 = time.time()
    result_list = []
    for run_ind in range(run_num):
        print(f'\n[run_ind={run_ind}]')
        result = retry_evaluate(dataset, metrics, column_map, retry_num=retry_num, retry_interval=retry_interval)
        result_list.append(result)

    multi_run_result = calcu_mean_var(result_list)
    t1 = time.time()
    print(f'evaluation time = {t1 - t0} s')
    return result_list, multi_run_result


def multi_evaluate_multi_dataset(
        dataset_list: Dataset,
        metrics: list[Metric] | None = None,
        column_map: dict[str, str] = {
            "question": "question",
            "contexts": "contexts",
            "answer": "answer",
            "ground_truths": "ground_truths",
        },
        retry_num=3,
        retry_interval=60 * 2,
) -> Tuple[List[Result], Dict]:
    print('run multi_evaluate_multi_dataset()')
    t0 = time.time()
    result_list = []
    for run_ind, dataset in enumerate(dataset_list):
        print(f'\n[run_ind={run_ind}]')
        result = retry_evaluate(dataset, metrics, column_map, retry_num=retry_num, retry_interval=retry_interval)
        result_list.append(result)

    multi_run_result = calcu_mean_var(result_list)
    t1 = time.time()
    print(f'evaluation time = {t1 - t0} s')
    return result_list, multi_run_result
