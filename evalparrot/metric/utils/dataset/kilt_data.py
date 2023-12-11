import json
import os

import requests
from langchain.schema import Document
from tqdm import tqdm
import difflib

try:
    from kilt.knowledge_source import KnowledgeSource
except:
    print('If you need to use KILT metric, please refer https://github.com/facebookresearch/KILT to setup KILT env.')

global_ks = KnowledgeSource()

# urls infos are from https://github.com/facebookresearch/KILT/tree/main
# exclude training jsonl
urls = [
    # "http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/fever-test_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/nq-test_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/hotpotqa-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/hotpotqa-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/hotpotqa-test_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/triviaqa-train_id-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/triviaqa-dev_id-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/triviaqa-test_id_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/eli5-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/eli5-test_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/trex-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/trex-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/trex-test_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-test_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/aidayago2-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/aidayago2-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/aidayago2-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/wned-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/wned-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/cweb-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/cweb-test_without_answers-kilt.jsonl",
    # "http://dl.fbaipublicfiles.com/KILT/wow-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/wow-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/wow-test_without_answers-kilt.jsonl",
]


def download_kilt_jsonl(kilt_data_path, kilt_dataset_name):
    data_urls = [url for url in urls if kilt_dataset_name in url]
    for url in data_urls:
        base = url.split("/")[-1]
        file_path = os.path.join(kilt_data_path, base)
        if os.path.exists(file_path):
            continue
        r = requests.get(url, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=base)
        with open(file_path, "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()


def find_data_jsonl_path(dataset_name, split, dataset_path):
    jsonl_path_list = []
    for file in os.listdir(dataset_path):
        if dataset_name in file and split in file and file.endswith('.jsonl'):
            jsonl_path_list.append(os.path.join(dataset_path, file))
    assert len(jsonl_path_list) == 1, 'num of found jsonl files should be 1.'
    return jsonl_path_list[0]


def _hash_provenance(wikipedia_id, start_paragraph_id, end_paragraph_id):
    return f'{wikipedia_id}_{start_paragraph_id}_{end_paragraph_id}'


def prepare_kilt_without_answer(kilt_data_path, kilt_dataset_name, split='dev', ks=global_ks):
    data_jsonl_path = find_data_jsonl_path(kilt_dataset_name, split, kilt_data_path)

    question_list = []
    id_list = []
    input_list = []
    # gt_answer_list = []  # no use, todo
    gt_contexts_list = []
    documents = []
    # all_datas = []
    wikipedia_id_set = set()
    provenance_hash_set = set()
    with open(data_jsonl_path, 'r', encoding="utf-8") as f:
        for line in tqdm(list(f)):
            line_dict = json.loads(line)
            # all_datas.append(line_dict)
            id_list.append(line_dict['id'])
            input_list.append(line_dict['input'])
            for one_of_output in line_dict['output']:
                if 'provenance' not in one_of_output:  # or 'answer' not in one_of_output:
                    continue
                question_list.append(line_dict['input'])
                # gt_answer_list.append(one_of_output['answer'])
                gt_contexts = []
                for one_of_provenance in one_of_output['provenance']:
                    wikipedia_id = one_of_provenance['wikipedia_id']
                    wikipedia_id_set.add(wikipedia_id)
                    page = ks.get_page_by_id(int(wikipedia_id))
                    assert page['wikipedia_id'] == page['_id']
                    start_paragraph_id = one_of_provenance['start_paragraph_id']
                    end_paragraph_id = one_of_provenance['end_paragraph_id']
                    start_character = one_of_provenance['start_character']
                    end_character = one_of_provenance['end_character']
                    assert start_paragraph_id == end_paragraph_id  # In KILT dataset, all start_paragraph_id equal end_paragraph_id
                    gt_contexts.append(page['text'][start_paragraph_id][start_character: end_character])
                    provenance_hash = _hash_provenance(wikipedia_id, start_paragraph_id, end_paragraph_id)
                    if provenance_hash in provenance_hash_set:
                        continue
                    else:
                        document = Document(page_content=page['text'][start_paragraph_id],
                                            metadata={'wikipedia_id': wikipedia_id,
                                                      'paragraph_id': start_paragraph_id,
                                                      })
                        documents.append(document)
                        provenance_hash_set.add(provenance_hash)
                gt_contexts_list.append(gt_contexts)
    documents.sort(key=lambda x: (int(x.metadata['wikipedia_id']),
                                  int(x.metadata['paragraph_id']),
                                  ))

    # question_list, gt_contexts_list are used for ragas
    # input_list, id_list are used for kilt
    # documents is used for insert docs into database
    assert len(question_list) == len(gt_contexts_list)
    assert len(input_list) == len(id_list)
    print(f'len(question_list) = len(gt_contexts_list) = {len(question_list)}')
    print(f'len(input_list) = len(id_list) = {len(input_list)}')
    print(f'len(documents) = {len(documents)}')
    return question_list, gt_contexts_list, documents, input_list, id_list


def prepare_kilt_without_answer_with_multi_documents(kilt_data_path, kilt_dataset_name, split='dev', pre_query_num=None,
                                                     ks=global_ks):
    data_jsonl_path = find_data_jsonl_path(kilt_dataset_name, split, kilt_data_path)

    question_list = []
    id_list = []
    input_list = []
    # gt_answer_list = []  # no use, todo
    gt_contexts_list = []
    documents = []
    # all_datas = []
    wikipedia_id_set = set()
    provenance_hash_set = set()
    if not pre_query_num:
        with open(data_jsonl_path, 'r', encoding="utf-8") as f:
            pre_query_num = len(list(f))
    with open(data_jsonl_path, 'r', encoding="utf-8") as f:
        for line in tqdm(list(f)[:pre_query_num]):
            line_dict = json.loads(line)
            # all_datas.append(line_dict)
            id_list.append(line_dict['id'])
            input_list.append(line_dict['input'])
            for one_of_output in line_dict['output']:
                if 'provenance' not in one_of_output:  # or 'answer' not in one_of_output:
                    continue
                question_list.append(line_dict['input'])
                # gt_answer_list.append(one_of_output['answer'])
                gt_contexts = []
                for one_of_provenance in one_of_output['provenance']:
                    wikipedia_id = one_of_provenance['wikipedia_id']
                    wikipedia_id_set.add(wikipedia_id)
                    page = ks.get_page_by_id(int(wikipedia_id))
                    assert page['wikipedia_id'] == page['_id']
                    start_paragraph_id = one_of_provenance['start_paragraph_id']
                    end_paragraph_id = one_of_provenance['end_paragraph_id']
                    start_character = one_of_provenance['start_character']
                    end_character = one_of_provenance['end_character']
                    assert start_paragraph_id == end_paragraph_id  # In KILT dataset, all start_paragraph_id equal end_paragraph_id
                    gt_contexts.append(page['text'][start_paragraph_id][start_character: end_character])

                gt_contexts_list.append(gt_contexts)

    # question_list, gt_contexts_list are used for ragas
    # input_list, id_list are used for kilt
    # documents is used for insert docs into database
    assert len(question_list) == len(gt_contexts_list)
    assert len(input_list) == len(id_list)
    print(f'len(question_list) = len(gt_contexts_list) = {len(question_list)}')
    print(f'len(input_list) = len(id_list) = {len(input_list)}')
    print(f'len(wikipedia_id_set) = {len(wikipedia_id_set)}')
    return question_list, gt_contexts_list, wikipedia_id_set, input_list, id_list


def filter_jsonl(data_jsonl_path, guess_output_path, filtered_data_jsonl_path, ks=global_ks):
    ids = set()
    with open(guess_output_path, 'r') as f1:
        for line in f1:
            data = json.loads(line)
            ids.add(data['id'])

    with open(data_jsonl_path, 'r') as f2, open(filtered_data_jsonl_path, 'w') as f3:
        for line in f2:
            data = json.loads(line)
            if data['id'] in ids:
                for one_of_output in data['output']:
                    if 'provenance' not in one_of_output:  # or 'answer' not in one_of_output:
                        continue
                    for one_of_provenance in one_of_output['provenance']:
                        wikipedia_id = one_of_provenance['wikipedia_id']
                        page = ks.get_page_by_id(int(wikipedia_id))
                        start_paragraph_id = one_of_provenance['start_paragraph_id']
                        src_context = page['text'][start_paragraph_id]
                        if 'meta' not in one_of_provenance:
                            one_of_provenance['meta'] = {}
                        one_of_provenance['meta']['src_context'] = src_context
                f3.write(json.dumps(data) + '\n')


# def find_common_substring_length(str1, str2):
#     max_length = 0
#     for i in range(len(str1)):
#         for j in range(len(str2)):
#             length = 0
#             m, n = i, j
#             while m < len(str1) and n < len(str2) and str1[m] == str2[n]:
#                 length += 1
#                 m += 1
#                 n += 1
#                 if length > max_length:
#                     max_length = length
#     return max_length

# def find_common_substring_length(str1, str2):
#     m = len(str1)
#     n = len(str2)
#     max_length = 0
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if str1[i - 1] == str2[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#                 max_length = max(max_length, dp[i][j])
#     return max_length


def find_common_substring_length(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    match = matcher.find_longest_match(0, len(str1), 0, len(str2))
    return match.size


def find_wikipedia_id_by_context(chunk_context, content_2_wikipedia_id):
    best_wikipedia_id = None
    max_score = 0
    best_context = None
    for context, wikipedia_id in content_2_wikipedia_id.items():
        score = find_common_substring_length(context, chunk_context)
        if score > max_score:
            max_score = score
            best_wikipedia_id = wikipedia_id
            best_context = context
    return best_wikipedia_id, best_context


def get_src_context_2_id(line_dict, ks=global_ks):
    context_2_id = dict()
    for one_of_output in line_dict['output']:
        for one_of_provenance in one_of_output['provenance']:
            wikipedia_id = one_of_provenance['wikipedia_id']
            page = ks.get_page_by_id(int(wikipedia_id))
            start_paragraph_id = one_of_provenance['start_paragraph_id']
            src_context = page['text'][start_paragraph_id]

            context_2_id[src_context] = one_of_provenance['wikipedia_id']

    return context_2_id


def find_wikipedia_id_by_gold(chunk_context, gold_line_dict, score_threshold=30, ks=global_ks):
    content_2_wikipedia_id = get_src_context_2_id(gold_line_dict, ks=ks)
    best_wikipedia_id = None
    max_score = 0
    best_context = None
    for context, wikipedia_id in content_2_wikipedia_id.items():
        score = find_common_substring_length(context, chunk_context)
        if score > max_score and score > score_threshold:
            max_score = score
            best_wikipedia_id = wikipedia_id
            best_context = context
    return best_wikipedia_id, best_context


def dump_wiki_doc(wikipedia_id, dst_path, ks=global_ks):
    page = ks.get_page_by_id(int(wikipedia_id))
    with open(dst_path, 'w') as f:
        for page_text in page['text']:
            if '::::' in page_text:
                f.write('\n')
            f.write(page_text)
