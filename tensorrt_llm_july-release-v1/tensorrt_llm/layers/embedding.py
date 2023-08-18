import math

from ..functional import embedding, unsqueeze, where
from ..module import Module
from ..parameter import Parameter


class Embedding(Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 dtype=None,
                 tp_size=1,
                 tp_group=None):
        super().__init__()
        # num_embeddings records the total vocab size no matter using TP or not
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_group = tp_group

        # When TP are involved (tp_size>1),
        # num_embeddings_tp is the size of the embedding numbers per process.
        self.weight = Parameter(shape=(math.ceil(
            self.num_embeddings / self.tp_size), self.embedding_dim),
                                dtype=dtype)
        self.tp_size = tp_size
        self.tp_group = tp_group

    def forward(self, x):
        return embedding(x,
                         self.weight.value,
                         tp_size=self.tp_size,
                         tp_group=self.tp_group)


class PromptTuningEmbedding(Embedding):
    """
        Pass all tokens though both normal and prompt embedding tables.
    Then, combine results based on whether the token was "normal" or "prompt/virtual".
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 vocab_size=None,
                 dtype=None,
                 tp_size=1,
                 tp_group=None):
        super().__init__(num_embeddings, embedding_dim, dtype, tp_size,
                         tp_group)
        if vocab_size is None:
            vocab_size = num_embeddings
        self.vocab_size = vocab_size

    def forward(self, tokens, prompt_embedding_table, tasks, task_vocab_size):
        # do not use ">=" because internally the layer works with floating points
        prompt_tokens_mask = tokens > (self.vocab_size - 1)

        # clip tokens in the [0, vocab_size) range
        normal_tokens = where(prompt_tokens_mask, self.vocab_size - 1, tokens)
        normal_embeddings = embedding(normal_tokens, self.weight.value,
                                      self.tp_size, self.tp_group)

        # put virtual tokens in the [0, max_prompt_vocab_size) range
        prompt_tokens = where(prompt_tokens_mask, tokens - self.vocab_size, 0)

        # add offsets to match the concatenated embedding tables
        tasks = tasks * task_vocab_size

        # tasks: [batch_size]
        # prompt_tokens: [batch_size, seq_len]
        prompt_tokens = prompt_tokens + unsqueeze(tasks, -1)
        prompt_embeddings = embedding(prompt_tokens, prompt_embedding_table)

        # prompt_tokens_mask: [batch_size, seq_len] -> [batch_size, seq_len, 1]
        # combine the correct sources of embedding: normal/prompt
        return where(unsqueeze(prompt_tokens_mask, -1), prompt_embeddings,
                     normal_embeddings)
