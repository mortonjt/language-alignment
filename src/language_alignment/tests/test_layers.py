import unittest
from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel, MeanAligner


class TestSSALayer(unittest.TestCase):

    def setUp(self):
        pass

    def test_ssa(self):
        pass


class TestCCALayer(unittest.TestCase):

    def setUp(self):
        pass

    def test_cca(self):
        pass


class TestFixedLayer(unittest.TestCase):

    def setUp(self):
        cls, path = pretrained_language_models['onehot']
        align_fun = MeanAligner()
        self.model = AlignmentModel(align_fun)
        self.model.load_language_model(cls, path, device='cpu')

    def test_fixed(self):
        self.model('ABCABCABCABC', 'SEQSEQSEQS')


if __name__ == '__main__':
    unittest.main()
