from ignite.engine import Engine
import torch


class WordSegmentationRNN(torch.nn.Module):
    """A pointwise word segmentation model.

    Args:
        num_char1 (int): Number of vocabulary in character unigram.
        num_char2 (int): Number of vocabulary in character bigram.
        num_char3 (int): Number of vocabulary in character trigram.
        char_dim (int): Dimensionality of character embeddings.
        num_chartype1 (int): Number of vocabulary in character type unigram.
        num_chartype2 (int): Number of vocabulary in character type bigram.
        num_chartype3 (int): Number of vocabulary in character type trigram.
        chartype_dim (int): Dimensionality of character type embeddings.
        hidden_size (int): Hidden size of LSTM.
        num_classes (int): Number of classes in the output layer.
        dropout (float, optional): Dropout probability.
    """

    def __init__(
        self,
        num_char1,
        num_char2,
        num_char3,
        char_dim,
        num_chartype1,
        num_chartype2,
        num_chartype3,
        chartype_dim,
        hidden_size,
        num_classes,
        dropout=0,
    ):
        super().__init__()
        self._num_char1 = num_char1
        self._num_char2 = num_char2
        self._num_char3 = num_char3
        self._char_dim = char_dim
        self._num_chartype1 = num_chartype1
        self._num_chartype2 = num_chartype2
        self._num_chartype3 = num_chartype3
        self._chartype_dim = chartype_dim
        self._wsize = 3
        self._ngram = 3
        self._input_size = 0
        for n in range(self._ngram):
            self._input_size += self._char_dim * (self._wsize * 2 - n)
            self._input_size += self._chartype_dim * (self._wsize * 2 - n)
        self._hidden_size = hidden_size
        self._num_layers = 2
        self._num_classes = num_classes
        self._dropout = dropout

        self._char1 = torch.nn.Embedding(self._num_char1, self._char_dim)
        self._char2 = torch.nn.Embedding(self._num_char2, self._char_dim)
        self._char3 = torch.nn.Embedding(self._num_char3, self._char_dim)
        self._chartype1 = torch.nn.Embedding(self._num_chartype1, self._chartype_dim)
        self._chartype2 = torch.nn.Embedding(self._num_chartype2, self._chartype_dim)
        self._chartype3 = torch.nn.Embedding(self._num_chartype3, self._chartype_dim)
        self._embedding_dropout = torch.nn.Dropout(self._dropout)
        self._lstm = torch.nn.LSTM(
            input_size=self._input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        )
        self._output = torch.nn.Linear(self._hidden_size, self._num_classes)

    def forward(self, char1, char2, char3, chartype1, chartype2, chartype3):
        """Forward pass.
        Args:
            char1 (torch.LongTensor, shape=[seq_len, batch_size, wsize]):
            char2 (torch.LongTensor, shape=[seq_len, batch_size, wsize-1]):
            char3 (torch.LongTensor, shape=[seq_len, batch_size, wsize-2]):
            chartype1 (torch.LongTensor, shape=[seq_len, batch_size, wsize]):
            chartype2 (torch.LongTensor, shape=[seq_len, batch_size, wsize-1]):
            chartype3 (torch.LongTensor, shape=[seq_len, batch_size, wsize-2]):
        """
        char1 = self._embedding_dropout(self._char1(char1))
        char1 = char1.reshape(char1.size(0), char1.size(1), -1)
        char2 = self._embedding_dropout(self._char2(char2))
        char2 = char2.reshape(char2.size(0), char2.size(1), -1)
        char3 = self._embedding_dropout(self._char3(char3))
        char3 = char3.reshape(char3.size(0), char3.size(1), -1)
        chartype1 = self._embedding_dropout(self._chartype1(chartype1))
        chartype1 = chartype1.reshape(chartype1.size(0), chartype1.size(1), -1)
        chartype2 = self._embedding_dropout(self._chartype2(chartype2))
        chartype2 = chartype2.reshape(chartype2.size(0), chartype2.size(1), -1)
        chartype3 = self._embedding_dropout(self._chartype3(chartype3))
        chartype3 = chartype3.reshape(chartype3.size(0), chartype3.size(1), -1)
        input_ = torch.cat((char1, char2, char3, chartype1, chartype2, chartype3), 2)
        output, hidden = self._lstm(input_)
        output = self._output(output)
        return output, hidden


class POSTaggingRNN(torch.nn.Module):
    """A pointwise part-of-speech tagging model.

    Args:
        num_char1 (int): Number of vocabulary in character unigram.
        num_char2 (int): Number of vocabulary in character bigram.
        num_char3 (int): Number of vocabulary in character trigram.
        char_dim (int): Dimensionality of character embeddings.
        num_chartype1 (int): Number of vocabulary in character type unigram.
        num_chartype2 (int): Number of vocabulary in character type bigram.
        num_chartype3 (int): Number of vocabulary in character type trigram.
        chartype_dim (int): Dimensionality of character type embeddings.
        num_word (int): Number of vocabulary in word.
        word_dim (int): Dimensionality of word embeddings.
        hidden_size (int): Hidden size of LSTM.
        num_classes (int): Number of classes in the output layer.
        dropout (float, optional): Dropout probability.
    """

    def __init__(
        self,
        num_char1,
        num_char2,
        num_char3,
        char_dim,
        num_chartype1,
        num_chartype2,
        num_chartype3,
        chartype_dim,
        num_word,
        word_dim,
        hidden_size,
        num_classes,
        dropout=0,
    ):
        super().__init__()
        self._num_char1 = num_char1
        self._num_char2 = num_char2
        self._num_char3 = num_char3
        self._char_dim = char_dim
        self._num_chartype1 = num_chartype1
        self._num_chartype2 = num_chartype2
        self._num_chartype3 = num_chartype3
        self._chartype_dim = chartype_dim
        self._num_word = num_word
        self._word_dim = word_dim
        self._wsize = 3
        self._ngram = 3
        self._input_size = self._word_dim
        for n in range(self._ngram):
            self._input_size += self._char_dim * (self._wsize * 2 - n)
            self._input_size += self._chartype_dim * (self._wsize * 2 - n)
        self._hidden_size = hidden_size
        self._num_layers = 2
        self._num_classes = num_classes
        self._dropout = dropout

        self._char1 = torch.nn.Embedding(self._num_char1, self._char_dim)
        self._char2 = torch.nn.Embedding(self._num_char2, self._char_dim)
        self._char3 = torch.nn.Embedding(self._num_char3, self._char_dim)
        self._chartype1 = torch.nn.Embedding(self._num_chartype1, self._chartype_dim)
        self._chartype2 = torch.nn.Embedding(self._num_chartype2, self._chartype_dim)
        self._chartype3 = torch.nn.Embedding(self._num_chartype3, self._chartype_dim)
        self._word = torch.nn.Embedding(self._num_word, self._word_dim)
        self._embedding_dropout = torch.nn.Dropout(self._dropout)
        self._lstm = torch.nn.LSTM(
            input_size=self._input_size,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        )
        self._output = torch.nn.Linear(self._hidden_size, self._num_classes)

    def forward(self, char1, char2, char3, type1, type2, type3, word):
        """Forward pass.

        Args:
            char1 (torch.LongTensor, shape=[seq_len, batch_size, 6]):
            char2 (torch.LongTensor, shape=[seq_len, batch_size, 5]):
            char3 (torch.LongTensor, shape=[seq_len, batch_size, 4]):
            type1 (torch.LongTensor, shape=[seq_len, batch_size, 6]):
            type2 (torch.LongTensor, shape=[seq_len, batch_size, 5]):
            type3 (torch.LongTensor, shape=[seq_len, batch_size, 4]):
            word (torch.LongTensor, shape=[seq_len, batch_size, 1]):
        """
        char1 = self._embedding_dropout(self._char1(char1))
        char1 = char1.reshape(char1.size(0), char1.size(1), -1)
        char2 = self._embedding_dropout(self._char2(char2))
        char2 = char2.reshape(char2.size(0), char2.size(1), -1)
        char3 = self._embedding_dropout(self._char3(char3))
        char3 = char3.reshape(char3.size(0), char3.size(1), -1)
        type1 = self._embedding_dropout(self._chartype1(type1))
        type1 = type1.reshape(type1.size(0), type1.size(1), -1)
        type2 = self._embedding_dropout(self._chartype2(type2))
        type2 = type2.reshape(type2.size(0), type2.size(1), -1)
        type3 = self._embedding_dropout(self._chartype3(type3))
        type3 = type3.reshape(type3.size(0), type3.size(1), -1)
        word = self._embedding_dropout(self._word(word))
        word = word.reshape(word.size(0), word.size(1), -1)
        input_ = torch.cat((char1, char2, char3, type1, type2, type3, word), 2)
        output, hidden = self._lstm(input_)
        output = self._output(output)
        return output, hidden


def _ws_train(model, loss_fn, optimizer, device, batch):
    model.train()
    optimizer.zero_grad()
    char1, char2, char3, chartype1, chartype2, chartype3, y = batch
    char1 = char1.to(device)
    char2 = char2.to(device)
    char3 = char3.to(device)
    chartype1 = chartype1.to(device)
    chartype2 = chartype2.to(device)
    chartype3 = chartype3.to(device)
    y = y.to(device)
    output, _ = model(char1, char2, char3, chartype1, chartype2, chartype3)
    loss = loss_fn(output.reshape(-1, output.size(2)), y.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def create_ws_trainer(model, loss_fn, optimizer, device):

    def _update(engine, batch):
        return _ws_train(model, loss_fn, optimizer, device, batch)

    return Engine(_update)


def _pt_train(model, loss_fn, optimizer, device, batch):
    model.train()
    optimizer.zero_grad()
    char1, char2, char3, chartype1, chartype2, chartype3, word, y = batch
    char1 = char1.to(device)
    char2 = char2.to(device)
    char3 = char3.to(device)
    chartype1 = chartype1.to(device)
    chartype2 = chartype2.to(device)
    chartype3 = chartype3.to(device)
    word = word.to(device)
    y = y.to(device)
    output, _ = model(char1, char2, char3, chartype1, chartype2, chartype3, word)
    loss = loss_fn(output.reshape(-1, output.size(2)), y.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def create_pt_trainer(model, loss_fn, optimizer, device):

    def _update(engine, batch):
        return _pt_train(model, loss_fn, optimizer, device, batch)

    return Engine(_update)
