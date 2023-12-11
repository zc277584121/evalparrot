import datetime
import json
import os
from typing import Union, List, Dict

from ragas.evaluation import Result


def save_result_to_csv(result: Result, csv_path: str):
    df = result.to_pandas()
    df.to_csv(csv_path)


def results_2_md_table(multi_run_result: Dict, method_name=None):
    table_head_str = f'''
| Metric              | {method_name}  |
|---------------------|----------------|
'''
    table_body_str = ''
    last_line = ''
    for metric_name, values in multi_run_result.items():
        now_line = f'|{metric_name} | {round(values["mean"], 2)}({round(values["var"], 4)}) |\n'
        if metric_name == 'ragas_score':
            last_line = now_line
        else:
            table_body_str = table_body_str + now_line
    table_str = table_head_str + table_body_str + last_line
    return table_str


def save_results(output_dir, result_name, result_list: Union[List[Result], Result], multi_run_result: Dict):
    if not isinstance(result_list, list):
        result_list = [result_list]
    result_dir = os.path.join(output_dir, result_name)
    for ind, result in enumerate(result_list):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        output_csv_path = os.path.join(result_dir, f'{result_name}_{ind}.csv')
        save_result_to_csv(result, output_csv_path)
    total_result_json_path = os.path.join(result_dir, f'{result_name}_total_result.json')
    total_result_dict = dict()
    total_result_dict['each_run_results'] = result_list
    total_result_dict['total_result'] = multi_run_result
    md_table_str = results_2_md_table(multi_run_result)
    print(md_table_str)
    with open(total_result_json_path, 'w') as f:
        f.write(json.dumps(total_result_dict, indent=2))
        print(f'save all results to {result_dir}')


def save_dataset_with_timestamp(dataset, output_dir, pre_name, save_csv=True, save_json=True):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    dataset.save_to_disk(os.path.join(output_dir, f'{pre_name}_{time_str}'))
    if save_csv:
        dataset.to_csv(os.path.join(output_dir, f'{pre_name}_{time_str}.csv'))
    if save_json:
        dataset.to_json(os.path.join(output_dir, f'{pre_name}_{time_str}.jsonl'))