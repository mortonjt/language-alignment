import unittest
from language_alignment import pretrained_language_models
from language_alignment.dataset import seq2onehot
from language_alignment.smith_waterman import smith_waterman_language_alignment


class TestSmithWatermanAlignment(unittest.TestCase):

    def setUp(self):
        pass

    def test_smith_waterman(self):
        exp_alignment = [
        '--------saQGAQIGAMLMAIRLRGmdleeTSVLTQALAQsgqqlewqseqlVPADGILYAARDVTAtvdSLPLITASILSKKLVEGl-----------'
        'dpvplpnvnaAILKKVIQWCTHHKDD-----PVWDQEFLKV-----------dQGTLFELILAANYLD---IKGLLDVTCKTVANMIKgktpeeirktfn'
        ]

        # X = seq2onehot(
        #     'SAQGAQIGAMLMAIRLRGMDLEETSVLTQALAQSGQQLEWQSEQLVPADGILYAARDVTATVDSLPLITASILSKKLVEGL'
        # )
        # Y = seq2onehot(
        #     'DPVPLPNVNAAILKKVIQWCTHHKDDPVWDQEFLKVDQGTLFELILAANYLDIKGLLDVTCKTVANMIKGKTPEEIRKTFN'
        # )
        X = seq2onehot('QGAQIGAMLMAIRLRG')
        Y = seq2onehot('AILKKVIQWCTHHKDD')
        arch = 'elmo'
        cls, path = pretrained_language_models[arch]
        lm = cls(path)
        sx = lm(X)
        sy = lm(Y)
        dm, score = smith_waterman_language_alignment(sx, sy)
        print(dm)
        print(score)


if __name__ == "__main__":
    unittest.main()
