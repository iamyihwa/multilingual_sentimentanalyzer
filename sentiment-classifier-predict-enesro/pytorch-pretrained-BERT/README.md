Contains training files and loading sentiment analyzer routine for the 
BERT (Bidirectional Encoder Representations from Transformer) 

It is based on the [hunggingface's pyTorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT) of original BERT. 

There are a couple of files that are useful.
(0) ./examples/run_classifier.py  need to change the labels of classes - currently changed to P, N, NEU, NONE

(1) Format_es_sentiment_input_pytorch.py changes the format of the input file so that it can be used in the current classifier.  

(2) In order to train follow steps in  StepsTrainSentimentClassifier-es.sh

(3) To predict with pretrained model, follow steps in the following notebook: Prediction_sentiment.ipynb

