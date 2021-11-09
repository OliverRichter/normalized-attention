import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Dense, LayerNormalization, Dropout
import numpy as np


class AddPositionalEmbedding(Layer):
    def __init__(self,
                 max_sequence_length,
                 sinosoidal=False,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 **kwargs):
        super(AddPositionalEmbedding, self).__init__(**kwargs)
        self._max_sequence_length = max_sequence_length
        self._sinosoidal = sinosoidal
        self._initializer = initializer

    def build(self, input_shape):
        if self._sinosoidal:
            # PE_(pos, 2i) = sin(pos / 10000 ^ (2i / d_model))
            # PE_(pos, 2i+1) = cos(pos / 10000 ^ (2i / d_model))
            position_enc = np.array([[pos / (10000 ** (2 * i / input_shape[-1])) for i in range(input_shape[-1])]
                                     if pos != 0 else np.zeros(input_shape[-1]) for pos in
                                     range(self._max_sequence_length)],
                                    dtype=np.float32)

            # index 0 is all zero
            position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # sine functions to 2i
            position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # cosine functions to 2i + 1
            self.pos_embedding = tf.constant(position_enc)
        else:
            self.pos_embedding = self.add_weight(shape=(self._max_sequence_length, input_shape[-1]),
                                                 initializer=self._initializer,
                                                 trainable=True)
        super(AddPositionalEmbedding, self).build(input_shape)

    def get_config(self):
        return {'max_sequence_length': self._max_sequence_length,
                'sinosoidal': self._sinosoidal,
                'initializer': self._initializer}

    def call(self, inputs):
        # slice positional embedding to the actual sequence length and add it
        return inputs + self.pos_embedding[:tf.shape(inputs)[1]]

    def compute_output_shape(self, input_shape):
        return input_shape


class AttentionLayer(Layer):
    def __init__(self,
                 number_of_heads,
                 query_dimension_total,
                 hidden_dimension_total=None,
                 output_dimension=None,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 regularizer=tf.keras.regularizers.l2(0.01),
                 logit_rescale=True,
                 attention_non_linearity=tf.nn.softmax,
                 attention_dropout_probability=0.1,
                 hidden_normalization=None,
                 hidden_non_linearity=None,
                 **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self._num_heads = number_of_heads
        self._querry_dim = query_dimension_total
        self._hidden_dim = hidden_dimension_total or query_dimension_total
        self._out_dim = output_dimension or query_dimension_total
        self._initializer = initializer
        self._regularizer = regularizer
        self._logit_rescale = logit_rescale
        if logit_rescale:
            # rescale the attention logits by division through the square root of the per head dimensionality
            self._logit_norm = lambda logit: logit / np.sqrt(float(self._querry_dim / self._num_heads))
        else:
            self._logit_norm = None
        self._att_non_lin = attention_non_linearity or tf.identity
        self._att_drop = attention_dropout_probability
        self._hidden_norm = hidden_normalization
        self._hidden_non_lin = hidden_non_linearity

    def get_config(self):
        return {'number_of_heads': self._num_heads,
                'query_dimension_total': self._querry_dim,
                'hidden_dimension_total': self._hidden_dim,
                'output_dimension': self._out_dim,
                'initializer': self._initializer,
                'regularizer': self._regularizer,
                'logit_rescale': self._logit_rescale,
                'attention_non_linearity': self._att_non_lin,
                'attention_dropout_probability': self._att_drop,
                'hidden_normalization': self._hidden_norm,
                'hidden_non_linearity': self._hidden_non_lin}

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.key_w = self.add_weight(shape=(input_shape[-1], self._querry_dim), initializer=self._initializer,
                                     trainable=True, regularizer=self._regularizer, name='key_weight')
        self.query_w = self.add_weight(shape=(input_shape[-1], self._querry_dim), initializer=self._initializer,
                                       trainable=True, regularizer=self._regularizer, name='query_weight')
        self.value_w = self.add_weight(shape=(input_shape[-1], self._hidden_dim), initializer=self._initializer,
                                       trainable=True, regularizer=self._regularizer, name='value_weight')
        self.key_b = self.add_weight(shape=(self._querry_dim,), initializer=tf.zeros_initializer(),
                                     trainable=True, name='key_bias')
        self.query_b = self.add_weight(shape=(self._querry_dim,), initializer=tf.zeros_initializer(),
                                       trainable=True, name='query_bias')
        self.value_b = self.add_weight(shape=(self._hidden_dim,), initializer=tf.zeros_initializer(),
                                       trainable=True, name='value_bias')
        self.mix_w = self.add_weight(shape=(self._hidden_dim, self._out_dim), initializer=self._initializer,
                                     trainable=True, regularizer=self._regularizer, name='mix_weight')
        self.mix_b = self.add_weight(shape=(self._out_dim,), initializer=tf.zeros_initializer(),
                                     trainable=True, name='mix_bias')
        super(AttentionLayer, self).build(input_shape)

    def _per_head(self, tensor):
        # reshape tensor into multiple heads
        sequence_length = tf.shape(tensor)[1]
        per_head_shape = (-1, sequence_length, self._num_heads, self._querry_dim // self._num_heads)
        tensor_reshaped = tf.reshape(tensor, per_head_shape)
        return tf.transpose(tensor_reshaped, [0, 2, 1, 3])  # position, head, value -> head, position, value

    def _concatenate_heads(self, tensor):
        # reshape tensor from multiple heads back to one embedding
        tensor_transposed = tf.transpose(tensor, [0, 2, 1, 3])  # head, position, value -> position, head, value
        sequence_length = tf.shape(tensor_transposed)[1]
        return tf.reshape(tensor_transposed, (-1, sequence_length, self._hidden_dim,))

    def call(self, inputs):
        inputs, mask = inputs
        keys = tf.matmul(inputs, self.key_w) + self.key_b
        queries = tf.matmul(inputs, self.query_w) + self.query_b
        values = tf.matmul(inputs, self.value_w) + self.value_b
        attention = tf.matmul(self._per_head(queries), self._per_head(keys), transpose_b=True)
        if self._logit_norm:
            attention = self._logit_norm(attention)
        attention = self._att_non_lin((attention, mask))
        if self._att_drop:
            attention = Dropout(self._att_drop)(attention)
        hidden = tf.matmul(attention, self._per_head(values))
        hidden = self._concatenate_heads(hidden)
        if self._hidden_norm:
            hidden = self._hidden_norm(hidden)
        if self._hidden_non_lin:
            hidden = self._hidden_non_lin(hidden)
        out = tf.matmul(hidden, self.mix_w) + self.mix_b
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self._out_dim,)


def rescale(inpt):
    # rescale by deviding through the square root of the sequence length
    sequence_length = tf.cast(tf.shape(inpt)[1], dtype=tf.float32)
    return inpt / tf.sqrt(sequence_length)


def normalize_masked(tensor, mask, axis=-1, epsilon=0.001):
    if mask is None:
        std = tf.math.reduce_std(tensor, axis=axis, keepdims=True) + epsilon
        return (tensor - tf.reduce_mean(tensor, axis=axis, keepdims=True)) / std
    counts = tf.reduce_sum(mask, axis=axis, keepdims=True)
    means = tf.reduce_sum(tensor * mask, axis=axis, keepdims=True) / counts
    centered = (tensor - means) * mask
    std = tf.sqrt(tf.reduce_sum(centered ** 2, axis=axis, keepdims=True) / counts) + epsilon
    return centered / std


class NormalizeAttention(Layer):
    def build(self, input_shape):
        number_of_heads = input_shape[0][1]  # input attention dimensions (batch, head, position_from, position_to)
        # use one gain and one bias value per head and broadcast over attention values and batch
        self._gain = self.add_weight(shape=(1, number_of_heads, 1, 1), initializer=tf.ones_initializer, trainable=True,
                                     name='gain')
        self._bias = self.add_weight(shape=(1, number_of_heads, 1, 1), initializer=tf.zeros_initializer, trainable=True,
                                     name='bias')

    def call(self, inputs):
        logits, mask = inputs
        return (self._gain * normalize_masked(logits, mask) + self._bias) * mask


def gelu(x):
    """
    Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    x: float Tensor to perform activation.
    Returns:
    `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class GeluLayer(Layer):
    def call(self, inputs):
        return gelu(inputs)


def ff_net(inputs,
           model_dimension,
           hidden_dimension,
           hidden_normalization=None,
           non_linearity_layer=GeluLayer,
           initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
           regularizer=tf.keras.regularizers.l2(0.01)):
    hidden = Dense(hidden_dimension,
                   activation=None,
                   kernel_initializer=initializer,
                   kernel_regularizer=regularizer)(inputs)
    if hidden_normalization:
        hidden = LayerNormalization()(hidden)
    if non_linearity_layer:
        hidden = non_linearity_layer()(hidden)
    out = Dense(model_dimension,
                activation=None,
                kernel_initializer=initializer,
                kernel_regularizer=regularizer)(hidden)
    return out


def create_model(vocabulary_size,
                 positional_embeddings,
                 max_sequence_length,
                 model,
                 model_dimension,
                 initial_std_factor,
                 layers,
                 heads,
                 dropout,
                 L2_regularization,
                 no_non_linearity_in_attention_block,
                 bert_layer_norm,
                 last_layer,
                 with_attention_mask,
                 **unused_params):

    print('init: ', initial_std_factor * 0.02)
    initializer = tf.keras.initializers.TruncatedNormal(stddev=initial_std_factor * 0.02)
    regularizer = tf.keras.regularizers.l2(L2_regularization)

    if model in ['NAP', 'NON', 'MTE', 'BERT']:
        ff_hidden_dimension = 4 * model_dimension
    else:
        # for sum and max pooling, the hidden layer in the feed forward networked is increased to approximately match
        # the parameter count of the attention models
        factor = (12.0 * model_dimension + 8) / (2 * model_dimension + 1)
        ff_hidden_dimension = round(factor * model_dimension)

    inputs = Input(shape=(None,), name='input_ids', dtype=tf.int64)
    if with_attention_mask:
        mask_input = Input(shape=(None,), name='attention_mask')
        attention_mask = mask_input[:, None, None, :]
        embeddings = Embedding(vocabulary_size, model_dimension, embeddings_initializer=initializer)(inputs)
        if positional_embeddings:
            embeddings = AddPositionalEmbedding(max_sequence_length, sinosoidal=True)(embeddings)
    else:
        attention_mask = tf.ones_like(inputs, dtype=tf.float32)[:, None, None, :]
        embeddings = Embedding(vocabulary_size, model_dimension, embeddings_initializer=initializer)(inputs)
        if positional_embeddings:
            embeddings = AddPositionalEmbedding(max_sequence_length)(embeddings)
    if dropout:
        embeddings = Dropout(dropout)(embeddings)

    for layer in range(layers):

        if model in ['NAP', 'NON', 'MTE', 'BERT', 'MTEgb']:
            hidden_normalization = None if bert_layer_norm else LayerNormalization()
            hidden_non_linearity = None if no_non_linearity_in_attention_block else GeluLayer()
            logit_rescale = not model == 'NAP'  # not needed in NAP due to normalization afterwards
            if model == 'NAP':
                attention_non_linearity = NormalizeAttention()
            elif model == 'NON':
                attention_non_linearity = lambda logits_and_mask: \
                    logits_and_mask[0] * logits_and_mask[1]
                hidden_normalization = rescale
            else:
                attention_non_linearity = lambda logits_and_mask: \
                    tf.nn.softmax(logits_and_mask[0] - (1.0 - logits_and_mask[1]) * 1.e9)

            pooling_out = AttentionLayer(number_of_heads=heads,
                                         query_dimension_total=model_dimension,
                                         initializer=initializer,
                                         regularizer=regularizer,
                                         logit_rescale=logit_rescale,
                                         attention_non_linearity=attention_non_linearity,
                                         attention_dropout_probability=dropout,
                                         hidden_normalization=hidden_normalization,
                                         hidden_non_linearity=hidden_non_linearity)((embeddings, attention_mask))
        elif model == 'w':
            print(embeddings.shape)
            embeddings_transposed = tf.transpose(embeddings, [0, 2, 1])
            embeddings_transposed = tf.ensure_shape(embeddings_transposed, [None, model_dimension, max_sequence_length])
            pooling_out = tf.transpose(Dense(max_sequence_length, activation=None,
                                             kernel_initializer=initializer,
                                             kernel_regularizer=regularizer)(embeddings_transposed), [0, 2, 1])
        elif model == 'sum':
            if with_attention_mask:
                embeddings = attention_mask[:, 0, 0, :, None] * embeddings
            pooling_out = tf.reduce_sum(embeddings, axis=1, keepdims=True)
        else:  # model == 'max':
            if with_attention_mask:
                embeddings -= (1.0 - attention_mask[:, 0, 0, :, None]) * 1.e9
            pooling_out = tf.reduce_max(embeddings, axis=1, keepdims=True)

        if bert_layer_norm:
            if dropout:
                pooling_out = Dropout(dropout)(pooling_out)
            pooling_out = LayerNormalization()(pooling_out + embeddings)
        else:
            pooling_out = LayerNormalization()(pooling_out)
            if dropout:
                pooling_out = Dropout(dropout)(pooling_out)
            pooling_out += embeddings

        ff_out = ff_net(pooling_out,
                        model_dimension,
                        ff_hidden_dimension,
                        hidden_normalization=not bert_layer_norm,
                        initializer=initializer,
                        regularizer=regularizer)

        if bert_layer_norm:
            if dropout:
                ff_out = Dropout(dropout)(ff_out)
            embeddings = LayerNormalization()(ff_out + pooling_out)
        else:
            ff_out = LayerNormalization()(ff_out)
            if dropout:
                ff_out = Dropout(dropout)(ff_out)
            embeddings = ff_out + pooling_out

    logits = last_layer(embeddings, initializer, regularizer)
    if with_attention_mask:
        return tf.keras.Model([inputs, mask_input], logits, name=model)
    return tf.keras.Model(inputs, logits, name=model)

