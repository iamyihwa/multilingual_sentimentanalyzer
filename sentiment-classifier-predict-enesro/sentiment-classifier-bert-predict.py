#on the terminal: 
#export FLASK_APP=flaskclassification.py
#flask run    (--host=0.0.0.0   this part is required when the process needs to be accessed from other computers )
    

from flask import Flask, request
import re


sent_labels = ["P", "N","NEU", "NONE"]

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["P", "N","NEU", "NONE"]
        #changed by yihwa 2019 05 07 
	#return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples





app = Flask(__name__)


from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

processor = ColaProcessor() 

label_list = processor.get_labels()
num_labels = len(label_list)

model_dir =  'model'  #'/tmp/es_3_sentiment_bert/'
out_dir = 'output'

model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case='false')



#Load in the text data and predict 
@app.route('/')
def hello_world():
    return '''<h3> Welcome to Taiger Multi Lingual Sentiment analysis service! </h3>
         <h3>"http://0.0.0.0:5401/enesro_sentiment_analyzer?text=your text" returns the result of sentiment classifier.</h3>'''


@app.route('/enesro_sentiment_analyzer')#, methods = ["POST"])
def query_sent_analyzer():
#	if request.method == 'POST':
	#import pandas as pd 
	#df_test = pd.read_csv('./data/test.csv', sep = '\t')
	#tagger: SequenceTagger = SequenceTagger.load_from_file('model/es-ner-glove.pt')
	text=request.args.get('text')


	import torch 
	max_seq_length = 128 


	tokens = tokenizer.tokenize(text)
	if len(tokens) > max_seq_length - 2:
    		tokens = tokens[:(max_seq_length - 2)]
	tokens = ["[CLS]"] + tokens + ["[SEP]"]
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	segment_ids = [0] * len(tokens)      
	input_mask = [1] * len(input_ids)  
	# Zero-pad up to the sequence length.
	padding = [0] * (max_seq_length - len(input_ids))
	input_ids += padding
	input_mask += padding
	segment_ids += padding	

	#change to pytorch tensor 

	input_ids = torch.tensor(input_ids , dtype = torch.long)
	#all_tokens = torch.tensor(tokens, dtype = torch.long)
	segment_ids = torch.tensor(segment_ids, dtype = torch.long)
	input_mask = torch.tensor(input_mask, dtype = torch.long)

	#change dimension so that there is a batch dimension 
	input_ids = input_ids.view(1, input_ids.size()[0])
	segment_ids = segment_ids.view(1, segment_ids.size()[0])
	input_mask = input_mask.view(1, input_mask.size()[0])
	logits = model(input_ids, segment_ids, input_mask, labels=None)
	

	softmax = torch.nn.functional.softmax(logits)
	conf, idx = torch.max(softmax, 0)
	#somehow returned idx were all 0s, need to check 


	predicted_label = sent_labels[conf.argmax()] 
	conf_val = conf.max().detach().numpy()
	print_label = " Label: %s (%.2f) " % (predicted_label, conf_val) 
	
	return '''{}'''.format(print_label)




if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
