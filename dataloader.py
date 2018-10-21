import _pickle as pk
import numpy as np

class Gen_Data_loader:

    def __init__(self, batch_size, SEQ_LENGTH):
        self.batch_size = batch_size
        self.token_stream = []
        self.SEQ_LENGTH = SEQ_LENGTH

    def create_batches(self, data_file):
        self.token_stream = []
        self.token_stream = pk.load(open(data_file, 'rb'))
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class F_Gen_Data_loader:

    def __init__(self, batch_size, SEQ_LENGTH):
        self.batch_size = batch_size
        self.token_stream = []
        self.SEQ_LENGTH = SEQ_LENGTH

    def create_batches(self, a, b):
        self.token_stream = []
        self.token_stream = pk.load(open(a, 'rb'))
        self.token_stream += pk.load(open(b, 'rb'))
        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader:

    def __init__(self, batch_size, SEQ_LENGTH):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.SEQ_LENGTH = SEQ_LENGTH

    def load_train_data(self, positive_file, negative_file):
        positive_examples = pk.load(open(positive_file, 'rb'))
        negative_examples = []
        with open(negative_file) as (fin):
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [ int(x) for x in line ]
                if len(parse_line) == self.SEQ_LENGTH:
                    negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)
        positive_labels = [ [1, 0] for _ in positive_examples ]
        negative_labels = [ [0, 1] for _ in negative_examples ]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = (
         self.sentences_batches[self.pointer], self.labels_batches[self.pointer])
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class F_Dis_dataloader:

    def __init__(self, batch_size, SEQ_LENGTH):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.SEQ_LENGTH = SEQ_LENGTH

    def load_train_data(self, positive_file_a, positive_file_b, negative_file):
        positive_examples_a = pk.load(open(positive_file_a, 'rb'))
        positive_examples_b = pk.load(open(positive_file_b, 'rb'))
        negative_examples = []
        with open(negative_file) as (fin):
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [ int(x) for x in line ]
                if len(parse_line) == self.SEQ_LENGTH:
                    negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples_a + positive_examples_b + negative_examples)
        positive_labels_a = [ [1, 0, 0] for _ in positive_examples_a ]
        positive_labels_b = [ [0, 1, 0] for _ in positive_examples_b ]
        negative_labels = [ [0, 0, 1] for _ in negative_examples ]
        self.labels = np.concatenate([positive_labels_a, positive_labels_b, negative_labels], 0)
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = (
         self.sentences_batches[self.pointer], self.labels_batches[self.pointer])
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class AB_Dis_dataloader:

    def __init__(self, batch_size, SEQ_LENGTH):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.SEQ_LENGTH = SEQ_LENGTH

    def load_train_data(self, positive_file_a, positive_file_b, negative_file, neutral_file):
        positive_examples_a = pk.load(open(positive_file_a, 'rb'))
        positive_examples_b = pk.load(open(positive_file_b, 'rb'))
        negative_examples = []
        neutral_examples = []
        with open(negative_file) as (fin):
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [ int(x) for x in line ]
                if len(parse_line) == self.SEQ_LENGTH:
                    negative_examples.append(parse_line)

        with open(neutral_file) as (fin):
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [ int(x) for x in line ]
                if len(parse_line) == self.SEQ_LENGTH:
                    neutral_examples.append(parse_line)

        self.sentences = np.array(positive_examples_a + positive_examples_b + negative_examples + neutral_examples)
        positive_labels_a = [ [1, 0, 0, 0] for _ in positive_examples_a ]
        positive_labels_b = [ [0, 1, 0, 0] for _ in positive_examples_b ]
        negative_labels = [ [0, 0, 1, 0] for _ in negative_examples ]
        neutral_labels = [ [0, 0, 0, 1] for _ in neutral_examples ]
        self.labels = np.concatenate([positive_labels_a, positive_labels_b, negative_labels, neutral_labels], 0)
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = (
         self.sentences_batches[self.pointer], self.labels_batches[self.pointer])
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0