import torch


def batchify_with_label(input_batch_list, gpu, padding_label):

    """
        input: list of words, chars and labels, various length. [[wordsid, labelsid],[wordsid,labelsid],...]
            words: word ids for one sentence. (batch_size, sent_len)
            labels: labels for one sentence.(batch_size, sent_len)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]#[[wordsid],[wordsid]...[wordsid]]
    labels = [sent[1] for sent in input_batch_list]#[[labelsid],[labelsid]...[labelsid]]
    word_seq_lengths = list(map(len, words))#得到batch中每个句子的长度
    max_seq_len = max(word_seq_lengths)
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    label_seq_tensor = torch.ones((batch_size, max_seq_len), requires_grad=False).long()
    label_seq_tensor = padding_label * label_seq_tensor
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)

    word_seq_lengths = torch.LongTensor(word_seq_lengths)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, label_seq_tensor, mask