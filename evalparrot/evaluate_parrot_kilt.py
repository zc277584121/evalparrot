import json
import os
import argparse

from kilt.eval_retrieval import evaluate
from kilt.knowledge_source import KnowledgeSource
from metric.utils.dataset.kilt_data import prepare_kilt_without_answer, find_data_jsonl_path, filter_jsonl, \
    prepare_kilt_without_answer_with_multi_documents, find_wikipedia_id_by_gold, download_kilt_jsonl

from metric.utils.io import save_dataset_with_timestamp, save_results
from metric.utils.parrot_utils.http_utils import post_create, post_delete, post_upsert_kilt, \
    post_upsert_kilt_with_multi_doc, post_search
from metric.utils.multi_run import multi_evaluate_one_dataset
from ragas.metrics import context_recall, context_precision  # , context_relevancy
from datasets import Dataset
from tqdm import tqdm

# dataset_with_paragraph_id = ['fever', 'triviaqa', 'wow', 'eli5', 'hotpotqa', 'nq', 'structured_zeroshot', 'trex']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kilt_dataset_name", type=str, required=False, default='hotpotqa')
    parser.add_argument("--kilt_wiki_mongo_domain", type=str, required=False, default='127.0.0.1')
    parser.add_argument("--parrot_service_address", type=str, required=False, default='http://127.0.0.1:8999')
    parser.add_argument("--milvus_domain", type=str, required=False, default='127.0.0.1')
    parser.add_argument("--result_name", type=str, required=False, default='kilt_parrot_evaluation_res')
    parser.add_argument("--top_k", type=int, required=False, default=10)
    parser.add_argument('--rerank', action='store_true')
    # parser.add_argument("--pre_answer_dataset", type=str, required=False, default=None)
    parser.add_argument("--pre_query_num", type=int, required=False, default=200)

    parser.add_argument('--metric_type', required=False, choices=['ragas_score', 'kilt_score'], default='kilt_score')
    parser.add_argument('--doc_gen_type', required=False, choices=['single', 'multi'], default='multi')

    args = parser.parse_args()

    metric_type = args.metric_type
    doc_gen_type = args.doc_gen_type
    pre_query_num = args.pre_query_num

    kilt_dataset_name = args.kilt_dataset_name
    project_name = args.result_name
    rerank = args.rerank if args.rerank is True else None
    top_k = args.top_k

    mongo_connection_string = f"mongodb://{args.kilt_wiki_mongo_domain}:27017/admin"
    knowledge_source = KnowledgeSource(mongo_connection_string=mongo_connection_string)

    output_dir = os.path.join('./outputs/kilt', project_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kilt_data_path = './datasets/kilt_data'
    if not os.path.exists(kilt_data_path):
        os.makedirs(kilt_data_path)
    download_kilt_jsonl(kilt_data_path, kilt_dataset_name)

    # if args.pre_answer_dataset:
    #     ds = Dataset.load_from_disk(args.pre_answer_dataset)
    # else:

    if doc_gen_type == 'multi':
        question_list, ground_truth_list, wikipedia_id_set, input_list, id_list = prepare_kilt_without_answer_with_multi_documents(
            kilt_data_path,
            kilt_dataset_name,
            split='dev',
            pre_query_num=pre_query_num,
            ks=knowledge_source
        )
    else:
        question_list, ground_truth_list, documents, input_list, id_list = prepare_kilt_without_answer(
            kilt_data_path,
            kilt_dataset_name,
            split='dev',
            ks=knowledge_source
        )
    if pre_query_num:
        print(f'use only pre {pre_query_num} number of data for query question.')
        question_list = question_list[:pre_query_num]
        ground_truth_list = ground_truth_list[:pre_query_num]
        input_list = input_list[:pre_query_num]
        id_list = id_list[:pre_query_num]
    else:
        print(f'use all data for query question.')
    temp_file_path = os.path.join(kilt_data_path, 'kilt_temp_doc.txt')

    try:
        post_delete(project_name=project_name, store_domain=args.milvus_domain,
                    parrot_domain=args.parrot_service_address, rerank=rerank)
    except:
        pass  # collection not exist, first time delete
    post_create(project_name=project_name, store_domain=args.milvus_domain, parrot_domain=args.parrot_service_address,
                rerank=rerank)
    if doc_gen_type == 'multi':
        post_upsert_kilt_with_multi_doc(project_name=project_name,
                                        ks=knowledge_source,
                                        wikipedia_id_set=wikipedia_id_set,
                                        temp_file_path=temp_file_path,
                                        store_domain=args.milvus_domain,
                                        parrot_domain=args.parrot_service_address,
                                        rerank=rerank)
    else:
        post_upsert_kilt(project_name=project_name,
                         kilt_dataset_name=kilt_dataset_name,
                         documents=documents,
                         temp_file_path=temp_file_path,
                         store_domain=args.milvus_domain,
                         parrot_domain=args.parrot_service_address,
                         rerank=rerank)

    contexts_list = []
    answer_list = []

    if metric_type == 'ragas_score':
        for question in tqdm(question_list):
            try:
                answer, contexts = post_search(
                    question,
                    project_name,
                    store_domain=args.milvus_domain,
                    parrot_domain=args.parrot_service_address,
                    top_k=top_k,
                    rerank=rerank
                )

            except Exception as e:
                print(e)
                answer = 'failed. please retry.'
                contexts = ['failed. please retry.']
                print('failed. please retry')
            contexts_list.append(contexts)
            answer_list.append(answer)
        ds = Dataset.from_dict({"question": question_list,
                                "contexts": contexts_list,
                                "answer": answer_list,
                                "ground_truths": ground_truth_list})

        # to huggingface dataset
        save_dataset_with_timestamp(ds, output_dir, pre_name='kilt_res', save_csv=True)

        result_list, multi_run_result = multi_evaluate_one_dataset(
            ds,
            metrics=[
                context_precision,
                context_recall,
            ],
            run_num=1,
        )
        save_results(output_dir, project_name, result_list, multi_run_result)
    else:
        data_jsonl_path = find_data_jsonl_path(kilt_dataset_name, 'dev', kilt_data_path)
        gold_id_2_dict = dict()
        with open(data_jsonl_path, 'r') as f:
            for line in f:
                gold_line_dict = json.loads(line)
                gold_id_2_dict[gold_line_dict['id']] = gold_line_dict
        guess_output_path = os.path.join(output_dir, 'guess_output.jsonl')
        with open(guess_output_path, 'w') as f:
            for ind, (input_, id_) in enumerate(tqdm(list(zip(input_list, id_list)))):
                if pre_query_num and ind > pre_query_num:
                    break
                try:
                    answer, contexts = post_search(
                        input_,
                        project_name,
                        store_domain=args.milvus_domain,
                        parrot_domain=args.parrot_service_address,
                        top_k=top_k,
                        rerank=rerank
                    )
                except Exception as e:
                    print(e)
                    answer = 'failed. please retry.'
                    contexts = ['failed. please retry.']
                    print('failed. please retry')
                id_ = id_list[ind]
                provenance = []
                for ctx_ind, chunk_context in enumerate(contexts):
                    wikipedia_id, src_context = find_wikipedia_id_by_gold(chunk_context.strip(), gold_id_2_dict[id_],
                                                                          ks=knowledge_source)
                    provenance_dict = {
                        "wikipedia_id": str(wikipedia_id),
                        "title": None,
                        "section": None,
                        "start_paragraph_id": None,  # int(paragraph_id),
                        "start_character": None,
                        "end_paragraph_id": None,  # int(paragraph_id),
                        "end_character": None,
                        "bleu_score": None,
                        'meta': {
                            'src_context': src_context,
                            'chunk_context': chunk_context
                        }
                    }
                    provenance.append(provenance_dict)
                output = [{
                    'answer': answer,
                    "provenance": provenance
                }]
                output_dict = {
                    'id': id_,
                    'input': input_,
                    'output': output,
                }
                one_line_res = json.dumps(output_dict)
                f.write(one_line_res + '\n')

        filtered_data_jsonl_path = os.path.join(output_dir, 'filtered_gold.jsonl')
        filter_jsonl(data_jsonl_path, guess_output_path, filtered_data_jsonl_path, ks=knowledge_source)
        eval_result = evaluate(gold=os.path.abspath(filtered_data_jsonl_path),
                               guess=os.path.abspath(guess_output_path),
                               ks=[1, 5],
                               rank_keys=['wikipedia_id']
                               )

        useful_keys = ['Rprec', 'precision@1', 'precision@5', 'recall@5', 'success_rate@5']
        eval_result = {key: value for key, value in eval_result.items() if key in useful_keys}
        output_json_path = os.path.join(output_dir, 'eval_result.json')
        with open(output_json_path, 'w') as fw:
            fw.write(json.dumps(eval_result, indent=4))
            print(f'save result to {output_json_path}')
