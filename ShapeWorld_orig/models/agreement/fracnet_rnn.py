from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters, fracnet_size, fracnet_depth, world_reduction, embedding_size, rnn, caption_reduction, multimodal_reduction, mlp_size, mlp_depth, soft):

    fracnet_sizes = [fracnet_size * 2**n for n in range(fracnet_depth)]
    if caption_reduction == 'state':
        rnn_state_size = fracnet_sizes[-1]
        rnn_size = None
    else:
        rnn_size = fracnet_sizes[-1]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=dataset_parameters['world_shape'], tensor=inputs.get('world')) >>
        FractalNet(sizes=fracnet_sizes) >>
        Reduction(reduction=world_reduction, axis=(1, 2))
    )

    caption = (
        (
            Input(name='caption', shape=dataset_parameters['caption_shape'], dtype='int', tensor=inputs.get('caption')) >>
            Embedding(indices=dataset_parameters['vocabulary_size'], size=embedding_size),
            Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
        ) >>
        Rnn(size=rnn_size, state_size=rnn_state_size, cell=rnn)
    )

    if caption_reduction == 'state':
        caption >>= Select(index=1)
    else:
        caption >>= Select(index=0) >> Reduction(reduction=caption_reduction, axis=1)

    agreement = (
        (world, caption) >>
        Reduction(reduction=multimodal_reduction) >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement
