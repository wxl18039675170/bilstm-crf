import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from model.crf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM_CRF(nn.Module):

    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        print("build batched BiLSTM CRF...")
        data.show_data_summary()
        self.embedding_dim = data.word_emb_dim
        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)

        # 声明embedding
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        # 将预训练词向量载入self.word_embeddings中
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        # 声明LSTM
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, lstm_hidden,
                            num_layers=self.lstm_layer, batch_first=True,
                            bidirectional=self.bilstm_flag)

        # 声明CRF
        self.index2label = {}
        for ele in data.label_alphabet.instance2index:
            self.index2label[data.label_alphabet.instance2index[ele]] = ele
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, len(self.index2label)+2)
        self.crf = CRF(len(self.index2label), data.HP_gpu)

        # 将模型载入到GPU中
        self.gpu = data.HP_gpu
        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lstm = self.lstm.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        """
        可以用来随机初始化word embedding
        """
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, batch_word, batch_wordlen):
        # batch_word: ([batch_size, max_sentence_length])
        # batch_wordlen: ([batch_size])
        embeds = self.word_embeddings(batch_word)
        # embeds: ([batch_size, max_word_length, embedding_dim])
        embeds = self.drop(embeds)
        #LSTM内置函数，确保每个batch能顺利运行，原理就是每个batch中的样本按长到短排序
        #batch_first，LSTM的输入是batch_size,句子长度，embedding_dim,如果是flase则为句子长度，batch_size
        embeds_pack = pack_padded_sequence(embeds, batch_wordlen, batch_first=True)
        #LSTM的输出
        out_packed, (h, c) = self.lstm(embeds_pack)
        lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True)
        # lstm_feature: ([batch_size, max_word_length, HP_hidden_dim])
        lstm_feature = self.droplstm(lstm_feature)
        lstm_feature = self.hidden2tag(lstm_feature)
        # lstm_feature: ([batch_size, max_word_length, len(self.index2label)+2])
        return lstm_feature

    def neg_log_likelihood(self, batch_word, mask, batch_label, batch_wordlen):
        """
        :param batch_word: ([batch_size, max_sentence_length])
        :param mask: ([batch_size, max_sentence_length])
        :param batch_label: ([batch_size, max_sentence_length])
        :param batch_wordlen: ([batch_size])
        :return:
        loss : 类似 tensor(3052.6426, device='cuda:0', grad_fn=<SubBackward0>)
        tag_seq: ([batch_size, max_sentence_length])
        """
        lstm_feature = self._get_lstm_features(batch_word, batch_wordlen)
        total_loss = self.crf.neg_log_likelihood_loss(lstm_feature, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(lstm_feature, mask)
        return total_loss, tag_seq

    def forward(self, batch_word, mask, batch_label, batch_wordlen):
        """
        :param batch_word: ([batch_size, max_sentence_length])
        :param mask: ([batch_size, max_sentence_length])
        :param batch_label: ([batch_size, max_sentence_length])
        :param batch_wordlen: ([batch_size])
        :return:
        tag_seq: ([batch_size, max_sentence_length])
        """
        lstm_feature = self._get_lstm_features(batch_word, batch_wordlen)
        scores, best_path = self.crf._viterbi_decode(lstm_feature, mask)
        return best_path
