# evalparrot

This is a tool to quantitatively test the performance of parrot.

## How to use?

- Step 1:
And install kilt, and it's dependency into pip:
```shell
git clone https://github.com/facebookresearch/KILT.git
cd KILT
conda create -n kilt37 -y python=3.7 && conda activate kilt37
pip install -e .
python setup.py install
```

- Step 2: Prepare Kilt mongodb datasource service:
refer to [kilt page](https://github.com/facebookresearch/KILT/tree/main?tab=readme-ov-file#kilt-knowledge-source). 


- Step 3:
Start parrot service


- Step 4:
Install evalparrot with it's dependencies:
```shell
pip install evalparrot
```

- Step 5:
Start evaluation:

```python
from evalparrot import evaluate_parrot_kilt

evaluate_parrot_kilt(
    kilt_dataset_name='hotpotqa',
    kilt_wiki_mongo_domain='127.0.0.1', # Use your mongo service address
    milvus_domain='127.0.0.1',  # Use the address of milvus in your parrot
    parrot_service_address='http://127.0.0.1:8999',  # Use the address of your parrot service
    result_name='kilt_parrot_evaluation_res',
    pre_query_num=10,
)
```

Parameter Description:
```text
Args:
    kilt_dataset_name (`str`):
        Available name include ['fever', 'triviaqa', 'wow', 'eli5', 'hotpotqa', 'nq', 'structured_zeroshot', 'trex'].
        Default is 'hotpotqa'.
    kilt_wiki_mongo_domain (`str`):
        You must first start the mongodb service of kilt datasource.
        Please refer to https://github.com/facebookresearch/KILT/tree/main?tab=readme-ov-file#kilt-knowledge-source
        Default is '127.0.0.1'.
    milvus_domain (`str`):
        Milvus service domain of parrot, default is '127.0.0.1'.
    parrot_service_address (`str`):
        Parrot service address, default is 'http://127.0.0.1:8999'.
    result_name (`str`):
        The name of the result, default is 'kilt_parrot_evaluation_res'.
    top_k (`int`):
        The parrot top k config, default is 10.
    rerank (`bool`):
        Whether to use rerank in parrot config, default is False.
    pre_query_num (`int`):
        The number of query questions is too large. You can only run the first n queries.
        Default is 200.
    metric_type (`str`):
        Available options include ['kilt_score', 'ragas_score'].
        Generally the default kilt score is used.
        When using 'ragas_score', you must set OPENAI_API_KEY in the environment variable,
        and this consumes a lot of openai token usage and can only measure some ragas scores without answers.
    doc_gen_type (`str`):
        Available options include ['multi', 'single'].
        'multi' is closer to the actual document settings,
        and the 'single' method comes from the baseline of ragas doc:
        https://github.com/explodinggradients/ragas/blob/main/experiments/baselines/fiqa/dataset-exploration-and-baseline.ipynb.
        Default is 'multi'.
```