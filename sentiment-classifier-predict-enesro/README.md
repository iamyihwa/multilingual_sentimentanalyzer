
This a sentiment classifier developed by finetuning [Multilingual Bert](https://github.com/google-research/bert/blob/master/multilingual.md). 

It was trained using English, Spanish  and Romanian sentiment data set.

Since the multilingual embedding is in 124 languages, the sentiment classifier could work on multiple other languages. Please feel free to try! 
 


__Sources of data__ 

__English__: [Source1](http://www.sepln.org/workshops/tass/tass_data/download.php?auth=NtQapsDsq45eTvZeZry) and [Source2](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)

 __Spanish__: [TASS2012](http://www.sepln.org/workshops/tass/2012/about.php), [TASS2016](http://www.sepln.org/workshops/tass/2016/tass2016.php) and [TASS2018](http://www.sepln.org/workshops/tass/2018/)
 
 __Romanian__: part of [IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) dataset translated into Romanian

## Prerequisites


## To install, follow these steps 
docker build -t image-sent-classifier-enesro:latest .

docker run -itd -p 5401:5000 --name sent-classifier-enesro image-sent-classifier-enesro

# Example usage: 
http://localhost:5401/enesro_sentiment_analyzer?text='imi place.'
http://localhost:5401/enesro_sentiment_analyzer?text='urasc asta.'
