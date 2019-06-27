# Project 2 - Identifying and Categorizing Offensive Language in Social Media 

## Description

This task requires the annotators to give their judgements on whether a tweet is offensive or not.

1. The annotators mark the tweet as being offensive or not offensive.
2. If the tweet is offensive then the annotators need to tell if the offense is targeted towards somebody or something or it is not targeted.
3. If the offense is targeted then the annotators also need to tell who it is targeted against.

## Data

Example:

Sub-task A: Offensive language identification

```txt
Hey @LIRR , you are disgusting. - Offensive
A true American literary icon. #PhilipRoth will be missed. - Not offensive
```

Sub-task B: Automatic categorization of offense types

```txt
I mean I'm dating to get fucking attention - Offensive Untargeted
Hey @LIRR , you are disgusting. - Offensive, Targeted Insult
```

Sub-task C: Offense target identification

```txt
@BreFields1 @jonesebonee18 fuck you lol - Offensive, Targeted Insult, Individual
@Top_Sergeant Assuming liberals are unarmed would be a grave mistake by the deplorables. - Offensive, Targeted Insult, Group
```

## Usage

```python
python3 main.py TRAIN_FILE TEST_FILE GOLD_FILE GLOVE_DIR MODEL
```

* `TRAIN_FILE`: path to training data (`olid-training-v1.0.tsv`)
* `TEST_FILE`: path to testing data (`testset-levela.tsv`)
* `GOLD_FILE`: path to gold standard labels (`labels-levela.csv`)
* `GLOVE_DIR`: directory to GloVe pre-trained embedding (`glove.twitter.27B`)
* `MODEL`: `CNN` or `RNN`

For example

```python
python3 main.py olid-training-v1.0.tsv testset-levela.tsv labels-levela.csv glove.twitter.27B RNN
```

## Methodology

### Preprocessing

* normalizing
  * tokens
  * hashtags
  * URLs
  * retweets (RT)
  * dates
  * elongated words (e.g., “Hiiiii” to “Hi”, partially hidden words (“c00l” to “cool”)
* converting emojis to text
* removing uncommon words
* using Twitter-specific tokenizers

Using TweetTokenizer from nltk

```python
from nltk.tokenize import TweetTokenizer
>>> tknzr = TweetTokenizer()
>>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
>>> tknzr.tokenize(s0)
['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']
```

### Pre-trained word vectors

GloVe train on Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors)

### Model

* CNN
* RNN (LSTM)

## Evaluation

macro-averaged F1-score

```python
sklearn.metrics.f1_score(y_true, y_pred, average='macro')
```

## Results

### Sub-task A

Method | F1-score
--- | ---
LSTM-GloVe(twitter.200d)-TweetTokenizer | 0.76672
CNN-GloVe(twitter.200d)-TweetTokenizer | 0.73247
CNN-GloVe(twitter.200d)-KerasTokenizer | 0.72209
CNN-GloVe(twitter.100d)-KerasTokenizer | 0.70459
ALL NOT | 0.41892
ALL OFF | 0.21818
