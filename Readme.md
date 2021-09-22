# Usage
This project is an implementation for the following paper: 
Conversational Graph Grounded Policy Learning for Open-Domain Conversation Generation
Jun Xu, Haifeng Wang, Zheng-Yu Niu, Hua Wu, Wanxiang Che, Ting Liu; ACL2020

Currently we have finished draw the what-vertex. You can directly get the trained what-vertex graph result at `test_output/20210917/result_18`. This result is trained on Persona data. 

To generate the conversational graph, we need to 1. collect the data(Weibo and Persona), 2. extract the keywords(Target-Guided Open-Domain Conversation), 3.Train the Multi-Mapping and Posterior Mapping Selection model. 4. Draw edges between the keywords(Generating Multiple Diverse Responses with Multi-Mapping and Posterior Mapping Selection). 

Persona Data: https://drive.google.com/file/d/1YGVGfEpW-UiVd8A24If3LYR1j9aHD13N/view?usp=sharing

Weibo Data: https://drive.google.com/file/d/1tW_AX8AhEmX-2UvO5kQG5CfU3TutLj0L/view?usp=sharing

## Data Preparation

This is the code for the following paper, and we use it for keywords extraction.

Target-Guided Open-Domain Conversation
Jianheng Tang, Tiancheng Zhao, Chenyan Xiong, Xiaodan Liang, Eric Xing, Zhiting Hu; ACL 2019

Requirement for Target-Guided Open-Domain Conversation

- `nltk==3.4`
- `tensoflow==1.12`
- `texar>=0.2.1 (Texar)`


First put the dialog corpus file into `preprocess/convai2/source`. The code has an extracted dialog corpus `all_none_original_no_cands.txt` from Persona in this folder.
(to change the dialog file , please refer to `preprocess/convai2/api.py`)
```shell
cd preprocess
python preprocess.py
```
The generated keywords which should be the original what-vertexes, are divied into three parts at `tx_data/train,valid,test/keywords.txt` . Copy the data to `IJCAI2019-MMPMS/data` .
(to change the keywords file location , please refer to `preprocess/prepare_data.py`)

## Data preprocess

This code is for trainning of MMPMS model and drawing edges between the keywords(what-vertexs).
This is an implementation of MMPMS model for the one-to-many problem in open-domain conversation. 
MMPMS employs a multi-mapping mechanism to capture the one-to-many responding regularities between an input post and its diverse responses with multiple mapping modules. 
MMPMS also incorporates a posterior mapping selection module to identify the mapping module corresponding to the target response for accurate optimization. 
Experiments on Weibo and Reddit conversation dataset demonstrate the capacity of MMPMS in generating multiple diverse and informative responses.For more details, see the IJCAI-2019 paper: Generating Multiple Diverse Responses with Multi-Mapping and Posterior Mapping Selection.

Requirement for IJCAI2019-MMPMS

- `Python >= 3.6`
- `PaddlePaddle >= 1.3.2 && <= 1.4.1` 
- `NLTK`

In the data file, each line is a post-response pair formatted by post \t response.

Training and inference configs are specified in the `run.py`.

Prepare pre-trained word embedding (e.g. sgns.weibo.300d.txt for Weibo and glove.840B.300d.txt for Reddit), and put it into the data folder. 
The first line of pre-trained word embedding file should be formatted by num_words embedding_dim.
For tranning of keywords mapping, we use glove.840B.300d.txt.

Preprocess the data by running:
```shell
python preprocess.py
```
Use the keywords file `data/train,valid,test.keywords.txt` to generate `.pkl` files for trainning.

## Train

To train the model , run:
```shell
python run.py --data_dir data --batch_size 2 --infer_batch_size 2 --save_dir $Save_DIR
```
Use the keywords file `data/train,valid,test.keywords.pkl` to train the model . An example model is located at `test_output/20210917/MMPMS-041609`
 
## Test

To obtain the connections between the what-vertexes , run:
```shell
python run.py --infer --model_dir MODEL_DIR --result_file RESULT_FILE
```
In addition , you can use the parameter `--num_mappings` to control the number of mappings.
The RESULT_FILE will be a Json file containing the input post, target response and predicted response from each mapping module.
An example result file is located at `test_output/20210917/result_18`, and this result is based on the Persona dataset .

If you want to see each neighbor node of what-vertex more intuitively, run:
```shell
python build_what_vertex.py
```
An example what_vertex mapping file is located at `test_output/20210917/result_18_what_vertex.txt`

## Note

When trainning and testing of the Multi-mapping model, be ware that the trainning parameters `--min_len` `--max_len` should be the same as inference parameters `--min_infer_len` `--max_infer_len`.
For keywords mapping, these four parameters are set to 1.
