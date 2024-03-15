import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.5

class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)
################################################################
''' GRU'''

<<<<<<< HEAD
=======
class ResNetGRU(nn.Module):
    def __init__(self, dropout_rate=0.5, pretrained=True):
        super(ResNetGRU, self).__init__()

        # Load a pretrained ResNet and remove the fully connected layer
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Identity()

        # Assuming resnet18 outputs 512 features
        resnet_features = 512

        # GRU Layer
        self.gru = nn.GRU(resnet_features, 128, 3, batch_first=True, dropout=dropout_rate)
        # self.gru = nn.GRU(resnet_features, 64, 3, batch_first=True, dropout=dropout_rate)
        # # Fully Connected Layer
        # self.fc = nn.Sequential(
        #     nn.Linear(64, 54),
        #     # nn.BatchNorm1d(54, eps=1e-05, momentum=0.2, affine=True),
        #     nn.LayerNorm(54),  # Apply LayerNorm
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout_rate),
        #     nn.Linear(54, 32),
        #     nn.LayerNorm(32),  # Apply LayerNorm
        #     # nn.BatchNorm1d(32, eps=1e-05, momentum=0.2, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 4)
        # )
        self.fc = nn.Sequential(
          nn.Linear(128, 64),
          nn.LayerNorm(64),  # Apply LayerNorm
          nn.ReLU(inplace=True),
          nn.Dropout(p=dropout_rate),
          nn.Linear(64, 32),
          nn.LayerNorm(32),  # Apply LayerNorm
          nn.ReLU(inplace=True),
          nn.Dropout(p=dropout_rate),
          nn.Linear(32, 16),
          nn.LayerNorm(16),  # Apply LayerNorm
          nn.ReLU(inplace=True),
          nn.Linear(16, 4)
        )

    def forward(self, x):
        # print("asdadsda\n\n")
        # print(x.size())
        x = x.repeat(1, 3, 1, 1)  # Repeats the channel dimension 3 times
        # print(x.size())
    # Directly use ResNet on the input x with shape (N, C, H, W)
        c_out = self.resnet(x)  # Output shape: (N, feature_size)

        # Introduce a sequence length dimension for GRU processing
        # After ResNet, reshape c_out to add a sequence length of 1: (N, 1, feature_size)
        r_out = c_out.unsqueeze(1)

        # GRU processing
        out, _ = self.gru(r_out)

        # Fully Connected Layer processing
        # Take the output of the last (and only) time step
        out = self.fc(out[:, -1, :])  # This simplifies to maintaining the shape (N, feature_size)

        return out
>>>>>>> a1bbad06d0b0e6c1e16f11dc14ab7b284e5d8555


################################################################
'''
LSTM
'''
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        #print("Input to EncoderLSTM shape:", x.shape) #Input to EncoderLSTM shape: torch.Size([32, 1, 22, 1000])
        x = x.squeeze(1)  # This removes the second dimension
        x = x.permute(0, 2, 1)  # Correctly reorder dimensions to LSTM's expected input format of (batch, seq, feature) = [32,1000,22]
        outputs, (hidden, cell) = self.lstm(x) 
        #print("EncoderLSTM outputs shape:", outputs.shape) #EncoderLSTM outputs shape: torch.Size([32, 1000, hidden])
        #print("EncoderLSTM hidden state shape:", hidden.shape)#EncoderLSTM hidden state shape: torch.Size([num_layer, 32, hidden])
        #print("EncoderLSTM cell state shape:", cell.shape)#EncoderLSTM cell state shape: torch.Size([num_layer, 32, hidden])
        return outputs, (hidden, cell)
        
        
class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(DecoderLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Ensure the LSTM layer's input size is set to hidden_size + output_size
        self.lstm = nn.LSTM(hidden_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        #print("Input to DecoderLSTMWithAttention shape:", input.shape) #Input to DecoderLSTMWithAttention shape: torch.Size([32, 1, hidden])
        attn_weights = self.attention(hidden[-1], encoder_outputs)#hidden state shape: torch.Size([num_layer, 32, hidden])
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)#attention weights size is:  torch.Size([32, 1000]) and #EncoderLSTM outputs shape: torch.Size([32, 1000, hidden])
        rnn_input = torch.cat((input, context), -1)
        #print("DecoderLSTMWithAttention concatenated input shape:", rnn_input.shape)#DecoderLSTMWithAttention concatenated input shape: torch.Size([32, 1, 2*hidden])
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        #print("DecoderLSTMWithAttention output shape:", output.shape) #DecoderLSTMWithAttention output shape: torch.Size([32, 1, hidden])
        output = self.fc(output.squeeze(1))
        #print("After FC layer output shape:", output.shape) #After FC layer output shape: torch.Size([32, 4])
        return output, hidden, cell, attn_weights
        
        
class Seq2SeqForClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(Seq2SeqForClassification, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderLSTMWithAttention(output_size, hidden_size, num_layers, dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, src, trg=None):
        #print("Input to Seq2Seq model shape:", src.shape) #Input to Seq2Seq model shape: torch.Size([32, 1, 22, 1000])
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = torch.zeros(src.size(0), 1, self.decoder.hidden_size).to(src.device)
        
        output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs) # output shape: torch.Size([32, 4])
        #output = self.fc(hidden[-1].squeeze(0)) #hidden [num_layer,32,hidden]
        output = self.fc(hidden[-1]) #hidden [num_layer,32,hidden]
        #print("Seq2Seq final output shape:", output.shape) #Seq2Seq final output shape: torch.Size([32, 4])
        return output


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        # Ensure hidden is from the last layer, shape: [batch_size, hidden_size]
        # if hidden.dim() == 3:  # multi-layer scenario
        #     hidden = hidden[-1]  # Take hidden state of the last layer
        
        #print('attention hidden size is: ',hidden.shape) #attention hidden size is:  torch.Size([32, hidden])
        hidden = hidden.unsqueeze(2)#[32, 16, 1] #encoder_outputs:torch.Size([32, 1000, 16])
        attn_weights = torch.bmm(encoder_outputs, hidden).squeeze(2)
        #print('attention weight size before softmax is: ',attn_weights.shape) #hidden size is:  torch.Size([32, hidden])
        #attention weight size before softmax is:  torch.Size([32, 1000])
        attn_weights = F.softmax(attn_weights, dim=1)
        #print('attention weight size after softmax is: ',attn_weights.shape) #hiden size is:  torch.Size([32, hidden])
        #attention weight size before softmax is:  torch.Size([32, 1000])
        return attn_weights



# class LSTM(nn.Module):
#     def __init__(self, dropout_rate=0.3):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(22, 64, 2, batch_first=True, dropout=dropout_rate)
#         self.fc = nn.Sequential(
            
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(num_features=32, eps=1e-05, momentum=0.2, affine=True),
#             nn.ReLU(inplace = True),
#             nn.Linear(32, 4)
#         )
    
#     def forward(self, x):
#         N, C, H, W = x.size()
#         print(x.size())
#         x = x.view(N, H, W).permute(0, 2, 1)
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out
########################################################








######################################################     
<<<<<<< HEAD
        







################################
=======
>>>>>>> a1bbad06d0b0e6c1e16f11dc14ab7b284e5d8555

##############################
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers
import tensorflow.keras.backend as K
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# fix CUDNN_STATUS_INTERNAL_ERROR
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
##########################################################

class AttentionLSTMIn(keras.layers.LSTM):
    """
    Keras LSTM layer with attention weights, extending Keras' LSTM layer.
    
    Attention weights are calculated based on the previous hidden state and the current input, 
    supporting both local and global attention mechanisms.
    """
    ATTENTION_STYLES = ['local', 'global']

    def __init__(self, units, alignment_depth=1, style='local', alignment_units=None, **kwargs):
        if style not in self.ATTENTION_STYLES:
            raise ValueError(f'Unrecognized style: {style}. Must be one of: {self.ATTENTION_STYLES}')

        self.alignment_depth = max(1, alignment_depth)
        self.alignment_units = [alignment_units or units] * self.alignment_depth if not isinstance(alignment_units, list) else alignment_units
        self.style = style
        #print('test passed')
        super(AttentionLSTMIn, self).__init__(units, **kwargs)

    def build(self, input_shape):
        if len(input_shape) <= 2:
            raise ValueError("Input shape must have more than 2 dimensions")
        
        self.samples, self.channels = input_shape[1:3]
        units_sequence = [self.units + (input_shape[-1] if self.style == 'local' else input_shape[1])]
        units_sequence += self.alignment_units + ([self.channels] if self.style == 'local' else [self.samples])
        
        self.attention_kernels = [self.add_weight(name=f'attention_kernel_{i}',
                                                  shape=(units_sequence[i], units_sequence[i + 1]),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
                                  for i in range(len(units_sequence) - 1)]
        
        self.attention_biases = [self.add_weight(name=f'attention_bias_{units_sequence[i + 1]}',
                                                 shape=(units_sequence[i + 1],),
                                                 initializer=self.bias_initializer,
                                                 regularizer=self.bias_regularizer,
                                                 constraint=self.bias_constraint)
                                 for i in range(len(units_sequence) - 1)] if self.use_bias else None
        super(AttentionLSTMIn, self).build(input_shape)

    def preprocess_input(self, inputs, training=None):
        self.input_tensor_hack = inputs
        return inputs

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory state
        energy = K.concatenate((inputs, h_tm1) if self.style == 'local' else (self.input_tensor_hack, K.repeat_elements(K.expand_dims(h_tm1), self.channels, -1)), axis=-1)
        if self.style == 'global':
            energy = K.permute_dimensions(energy, (0, 2, 1))

        for i, kernel in enumerate(self.attention_kernels):
            energy = K.dot(energy, kernel)
            if self.use_bias:
                energy = K.bias_add(energy, self.attention_biases[i])
            energy = self.activation(energy)

        alpha = K.softmax(energy)
        
        if self.style == 'local':
            inputs *= alpha
        else:
            alpha = K.permute_dimensions(alpha, (0, 2, 1))
            inputs = K.sum(self.input_tensor_hack * alpha, axis=1)

        return super(AttentionLSTMIn, self).step(inputs, states)




def RaSCNN(inputshape, outputshape, params=None):
    """
  
    Adapted from https://github.com/cmunozcortes/c247-final-project/tree/main
    """
    ret_seq = True
    att_depth = 4
    attention = 76
    temp_layers = 4
    steps = 2
    temporal = 24
    temp_pool = 20

    lunits = [200, 40]
    activation = keras.activations.selu
    reg = float(0.01)
    dropout = 0.55

    convs = [inputshape[-1]//steps for _ in range(1, steps)]
    convs += [inputshape[-1] - sum(convs) + len(convs)]

    ins = keras.layers.Input(inputshape)
    conv = keras.layers.Reshape((inputshape[0], inputshape[1], 1))(ins)

    for i, c in enumerate(convs):
        conv = keras.layers.Conv2D(lunits[0]//len(convs),
            (1, c), activation=activation,
            name='spatial_conv_{0}'.format(i),
            kernel_regularizer=tf.keras.regularizers.l2(reg))(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.SpatialDropout2D(dropout)(conv)

    for i in range(temp_layers):
        conv = keras.layers.Conv2D(lunits[1], (temporal, 1), activation=activation,
            use_bias=False, name='temporal_conv_{0}'.format(i),
            kernel_regularizer=tf.keras.regularizers.l2(reg))(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.AveragePooling2D((temp_pool, 1,))(conv)
    conv = keras.layers.SpatialDropout2D(dropout)(conv)
    conv = keras.layers.Reshape((45, 40))(conv)

    attn = keras.layers.Bidirectional(AttentionLSTMIn(attention,
        implementation=2,
        dropout=dropout,
        return_sequences=ret_seq,
        alignment_depth=att_depth,
        style='global',
        kernel_regularizer=tf.keras.regularizers.l2(reg),
        ))(conv)
    conv = keras.layers.BatchNormalization()(attn)

    if ret_seq:
        conv = keras.layers.Flatten()(conv)
    outs = conv
    for units in lunits[2:]:
        outs = keras.layers.Dense(units, activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(reg))(outs)
        outs = keras.layers.BatchNormalization()(outs)
        outs = keras.layers.Dropout(dropout)(outs)
    outs = keras.layers.Dense(outputshape, activation='softmax')(outs)

    return keras.models.Model(ins, outs)





<<<<<<< HEAD
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.regularizers import l2


# def RaSCNN(input_shape, output_shape, att_depth=4, params=None):
#     """
#     Spatial summary CNN augmented with attention-focused recurrence (Ra-SCNN).
#     """
#     ret_seq = True
#     activation = keras.activations.selu
#     reg_rate = 0.01
#     dropout_rate = 0.55
#     spatial_units = 200
#     temporal_units = 40
#     attention_units = 76
#     temporal_kernel_size = 24
#     temporal_pool_size = 20
#     temp_layers = 4
#     steps = 2

#     conv_splits = [input_shape[-1] // steps for _ in range(steps)]
#     conv_splits[-1] += input_shape[-1] - sum(conv_splits)

#     inputs = keras.layers.Input(input_shape)
#     conv = keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)

#     for i, conv_size in enumerate(conv_splits):
#         num_filters = spatial_units // len(conv_splits)
#         conv = keras.layers.Conv2D(num_filters, (1, conv_size), activation=activation,
#                                   name='spatial_conv_{0}'.format(i),
#                                   kernel_regularizer=keras.regularizers.l2(reg_rate))(conv)
#     conv = keras.layers.BatchNormalization()(conv)
#     conv = keras.layers.SpatialDropout2D(dropout_rate)(conv)

#     for i in range(temp_layers):
#         conv = keras.layers.Conv2D(temporal_units, (temporal_kernel_size, 1), activation=activation,
#                                   use_bias=False, name='temporal_conv_{0}'.format(i),
#                                   kernel_regularizer=keras.regularizers.l2(reg_rate))(conv)
#     conv = keras.layers.BatchNormalization()(conv)
#     conv = keras.layers.AveragePooling2D((temporal_pool_size, 1))(conv)
#     conv = keras.layers.SpatialDropout2D(dropout_rate)(conv)
#     conv = keras.layers.Reshape((-1, temporal_units))(conv)

#     attn = keras.layers.Bidirectional(AttentionLSTMIn(attention_units,
#                                                       implementation=2,
#                                                       dropout=dropout_rate,
#                                                       return_sequences=ret_seq,
#                                                       alignment_depth=att_depth,
#                                                       style='global',
#                                                       kernel_regularizer=keras.regularizers.l2(reg_rate),
#                                                       ))(conv)
#     conv = keras.layers.BatchNormalization()(attn)

#     if ret_seq:
#         conv = keras.layers.Flatten()(conv)
#     outs = conv

#     for units in lunits[2:]::  # Assuming lunits[2:] was intended to be here, adjust as needed.
#         outs = keras.layers.Dense(units, activation=activation,
#                                   kernel_regularizer=keras.regularizers.l2(reg_rate))(outs)
#         outs = keras.layers.BatchNormalization()(outs)
#         outs = keras.layers.Dropout(dropout_rate)(outs)
#     outs = keras.layers.Dense(output_shape, activation='softmax')(outs)

#     return keras.models.Model(inputs, outs)


=======
>>>>>>> a1bbad06d0b0e6c1e16f11dc14ab7b284e5d8555

    
