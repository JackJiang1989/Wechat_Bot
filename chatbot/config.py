""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This is based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

"""

# parameters for processing the dataset
DATA_PATH = '/home/jackjiang/chatbot/data/test'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

#TESTSET_SIZE should be smaller than training lines
TESTSET_SIZE = 25000

BUCKETS = [(6, 8), (10, 12), (14, 16)]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0
