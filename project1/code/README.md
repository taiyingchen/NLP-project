# Project 1 - Fake News Detection

## Description

Classification of news article

Given the title of a fake news article A and the title of a coming news article B, students are asked to classify B into one of the three categories.

* agreed: B talks about the same fake news as A
* disagreed: B refutes the fake news in A
* unrelated: B is unrelated to A

Kaggle link: <https://www.kaggle.com/c/fake-news-pair-classification-challenge>

## Data

Example:

```
A: 用大蒜鉴别地沟油的方法,怎么鉴别地沟油
B: 吃了30年食用油才知道，一片大蒜轻松鉴别地沟油
Agreed
A: "飞机就要起飞，一个男人在机舱口跪下！"这是最催泪的一幕	
B:「网警辟谣」飞机起飞前男人机舱口跪下？这故事居然是编的！
Disagreed
A: 吃榴莲的禁忌,吃错会致命!
B: 榴莲不能和什么一起吃 与咖啡同吃诱发心脏病
Unrelated
```

## Installation

```bash
pip install -r requirements.txt
```

### Packages

```txt
bert-tensorflow
tensorflow
tensorflow-hub
nltk
sklearn
pandas
```

## Usage

### Minimum Edit Distance

#### Train

No need to train.

#### Test

```bash
bash med_test.sh path/to/test/csv path/to/prediction/csv
```

For example

```bash
bash med_test.sh test.csv pred.csv
```

### BERT

#### Train

```bash
bash bert_train.sh path/to/train/csv path/to/output/model/dir
```

For example

```bash
bash bert_train.sh train.csv model/
```

#### Test

```bash
bash bert_test.sh path/to/test/csv path/to/model/dir path/to/prediction/csv
```

For example

```bash
bash bert_test.sh test.csv model/ pred.csv
```

## Results

Method | Private score | Public score
--- | ---| ---
All unrelated | 0.57842 | 0.66675
Min. edit distance (T = 20.7782) | 0.70500 | 0.70956
BERT | 0.85879 | 0.86242
