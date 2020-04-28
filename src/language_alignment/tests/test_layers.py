import unittest
from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import MeanAligner, SSAaligner, CCAaligner


class TestSSAaligner(unittest.TestCase):

    def setUp(self):
        cls, path = pretrained_language_models['onehot']
        align_fun = SSAaligner()
        self.model = AlignmentModel(align_fun)
        self.model.load_language_model(cls, path, device='cpu')

    def test_ssa(self):
        self.model('ABCABCABCABC', 'SEQSEQSEQS')


class TestCCAaligner(unittest.TestCase):

    def setUp(self):
        cls, path = pretrained_language_models['onehot']
        align_fun = CCAaligner(input_dim=22, embed_dim=5, max_len=10)
        self.model = AlignmentModel(align_fun)
        self.model.load_language_model(cls, path, device='cpu')

    def test_cca(self):
        self.model('ABCABCABCA', 'SEQSEQSEQS')


class TestMeanAligner(unittest.TestCase):

    def setUp(self):
        cls, path = pretrained_language_models['onehot']
        align_fun = MeanAligner()
        self.model = AlignmentModel(align_fun)
        self.model.load_language_model(cls, path, device='cpu')

    def test_fixed(self):
        self.model('ABCABCABCABC', 'SEQSEQSEQS')


if __name__ == '__main__':
    unittest.main()
