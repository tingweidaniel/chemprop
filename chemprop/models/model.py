from argparse import Namespace

import torch.nn as nn
import torch

from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                TimeDistributed_wrapper(dropout),
                TimeDistributed_wrapper(nn.Linear(first_linear_dim, args.output_size, bias=False))
            ]
        else:
            ffn = [
                TimeDistributed_wrapper(dropout),
                TimeDistributed_wrapper(nn.Linear(first_linear_dim, args.ffn_hidden_size, bias=False))
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    TimeDistributed_wrapper(activation),
                    TimeDistributed_wrapper(dropout),
                    TimeDistributed_wrapper(nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size, bias=False)),
                ])
            ffn.extend([
                TimeDistributed_wrapper(activation),
                TimeDistributed_wrapper(dropout),
                TimeDistributed_wrapper(nn.Linear(args.ffn_hidden_size, args.output_size, bias=False)),
                LambdaLayer(lambda x: torch.sum(x, 1))  
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass')
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model

class LambdaLayer(nn.Module):
    def __init__(self, lambda_function):
        super(LambdaLayer, self).__init__()
        self.lambda_function = lambda_function
    def forward(self, x):
        return self.lambda_function(x)
    

class TimeDistributed_wrapper(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed_wrapper, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
