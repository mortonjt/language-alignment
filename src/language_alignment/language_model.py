import os
import torch
import warnings
from language_alignment.protein import ProteinSequence
from language_alignment.util import onehot
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf
from tape import ProteinBertModel, UniRepModel
from allennlp.commands.elmo import ElmoEmbedder


class LanguageModel(torch.nn.Module):
    def __init__(self, path=None):
        super(LanguageModel, self).__init__()
        self.path = path

    def __call__(self, x):
        pass


class Unirep(LanguageModel):
    def __init__(self, path=None):
        super(Unirep, self).__init__()
        self.model = UniRepModel.from_pretrained('babbler-1900')

    def __call__(self, x):
        output = self.model(x)
        print(output[0].shape)
        print(output[1].shape)
        return output[0].permute(0, 2, 1)


class Seqvec(LanguageModel):
    def __init__(self, path=None):
        super(Seqvec, self).__init__()
        weights_path = f'{path}/weights.hdf5'
        options_path = f'{path}/options.json'
        self.model = ElmoEmbedder(
            weight_file=str(weights_path),
            options_file=str(options_path)).elmo_bilm

    def __call__(self, x):
        output = self.model(x)['activations'][-1]
        embed = output.view(
            output.shape[0], output.shape[1] * output.shape[2]
        )
        embed = embed.unsqueeze(0).permute(0, 2, 1)
        return embed


class Bert(LanguageModel):
    def __init__(self, path=None):
        super(Bert, self).__init__()
        self.model = ProteinBertModel.from_pretrained('bert-base')

    def __call__(self, x):
        output = self.model(x)
        return output[0].permute(0, 2, 1)


class Roberta(LanguageModel):
    def __init__(self, path, trainable=False):
        super(Roberta, self).__init__(path)
        from fairseq.models.roberta import RobertaModel
        ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        # can substitute any model for `checkpoint_xs.pt`
        self.model = RobertaModel.from_pretrained(
            path, 'checkpoint_best.pt',
            gpt2_encoder_json=f'{ROOT}/peptide_bpe/encoder.json',
            gpt2_vocab_bpe=f'{ROOT}/peptide_bpe/vocab.bpe')

    def __call__(self, x):
        """ Extracts representation from one hot encodings. """
        #toks = self.model.encode(' '.join(list(x)))
        res = self.model.extract_features(x)
        # cut out the ends
        res = res[:, 1:-1]
        return res.permute(0, 2, 1)


class Elmo(LanguageModel):
    # requires a GPU in order to test
    def __init__(self, path, trainable=False, device='cuda',
                 per_process_gpu_memory_fraction=0.2):
        super(Elmo, self).__init__(path, device)
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        m = tf.keras.models.load_model(path)
        layer = 'LSTM2'
        self.model = tf.keras.models.Model(inputs=[m.input],
                                           outputs=[m.get_layer(layer).output],
                                           trainable=trainable)

    def __call__(self, x):
        prot = ProteinSequence(x)
        embed = self.model.predict(prot.onehot).squeeze()
        return torch.Tensor(embed)


class OneHot(LanguageModel):
    def __init__(self, path):
        super(OneHot, self).__init__(path)
        self.tla_codes = ["A", "R", "N", "D", "B", "C", "E", "Q", "Z", "G",
                          "H", "I", "L", "K", "M", "F", "P", "S", "T", "W",
                          "Y", "V"]
        self.num_words = len(self.tla_codes)

    def __call__(self, x):
        emb_i = [onehot(self.tla_codes.index(w_i), self.num_words) for w_i in x]

        emb = torch.Tensor(emb_i).t()
        x, y = emb.shape
        emb = emb.view(1, x, y)
        # make the embedding dimension the last dimension
        return emb
