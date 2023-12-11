#### This py is used for new parrot version, main branch

import os
import requests

# URL_DOMAIN = 'http://127.0.0.1:8998'
from langchain.schema import Document
from tqdm import tqdm

from ..dataset.kilt_data import dump_wiki_doc

PARROT_DOMAIN = 'http://127.0.0.1:8999'
STORE_DOMAIN = 'http://127.0.0.1'


DB_CONFIG = {
  "db_id": 'test-001',
  "lang": "en",
  "chunk_limit": -1,
  "embedding": "bge-base",
  "rerank": None,
  "milvus": {'uri': f"{STORE_DOMAIN}:19530", 'token': None}
}

def build_db_config(kb_id, rerank, store_domain):
    db_config = DB_CONFIG.copy()
    db_config['kb_id'] = kb_id
    db_config['rerank'] = rerank
    db_config['milvus']['uri'] = f"http://{store_domain}:19530"
    return db_config


search_param = {
    "top_k": 10,
    "offset": 0,
    "output_fields": [
      "doc_name",
      "chunk_id",
      "chunk_text"
    ],
    "expr": None
}

def post_delete(project_name, store_domain=STORE_DOMAIN, parrot_domain=PARROT_DOMAIN, rerank=None):
    url = parrot_domain + '/api/v1/db/delete'
    db_config = build_db_config(project_name, rerank, store_domain)
    response = requests.post(url, json=db_config)
    assert response.status_code == 200


def post_create(project_name, store_domain=STORE_DOMAIN, parrot_domain=PARROT_DOMAIN, rerank=None):
    url = parrot_domain + '/api/v1/db/create'
    db_config = build_db_config(project_name, rerank, store_domain)
    response = requests.post(url, json={'db_config': db_config})
    assert response.status_code == 200
    # print(response)


def post_upsert_kilt(project_name, kilt_dataset_name, documents, temp_file_path, store_domain=STORE_DOMAIN, parrot_domain=PARROT_DOMAIN, rerank=None):
    content_2_wikipedia_id = dict()
    with open(temp_file_path, 'w') as f:
        for document in tqdm(documents):
            content_2_wikipedia_id[document.page_content.strip()] = document.metadata['wikipedia_id']
            f.write(document.page_content)
    temp_file_path = os.path.abspath(temp_file_path)
    url = parrot_domain + '/api/v1/document/upsert'
    db_config = build_db_config(project_name, rerank, store_domain)
    response = requests.post(url, json={
        'doc_name': kilt_dataset_name,
        'source': temp_file_path,
        'db_config': db_config
    })
    assert response.status_code == 200
    token_used = response.json()['data']
    assert isinstance(token_used, int) and token_used > 0
    return content_2_wikipedia_id

def post_upsert_kilt_with_multi_doc(project_name, ks, wikipedia_id_set, temp_file_path, store_domain=STORE_DOMAIN, parrot_domain=PARROT_DOMAIN, rerank=None):
    for wikipedia_id in tqdm(list(wikipedia_id_set)):
        dump_wiki_doc(wikipedia_id, temp_file_path, ks=ks)
        temp_file_path = os.path.abspath(temp_file_path)
        url = parrot_domain + '/api/v1/document/upsert'
        db_config = build_db_config(project_name, rerank, store_domain)
        response = requests.post(url, json={
            'doc_name': wikipedia_id,
            'source': temp_file_path,
            'db_config': db_config
        })
        assert response.status_code == 200
        token_used = response.json()['data']
        assert isinstance(token_used, int) and token_used > 0


def post_search(query, project_name, store_domain=STORE_DOMAIN, parrot_domain=PARROT_DOMAIN, top_k=None, rerank=None):
    url = parrot_domain + '/api/v1/search'
    db_config = build_db_config(project_name, rerank, store_domain)
    search_param['top_k'] = top_k
    post_json = {
        'query': query,
        'db_config': db_config,
        'search_param': search_param
    }
    response = requests.post(url, json=post_json)
    assert response.status_code == 200
    result_list = response.json()['data']
    contexts = [res['chunk_text'] for res in result_list]
    answer = 'no answer.' # mock answer
    return answer, contexts




