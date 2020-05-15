import torch
import unittest
from language_alignment.dataset.dataset import seq2onehot
from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import CCAaligner
from tape import TAPETokenizer
import numpy as np
import numpy.testing as npt


class TestLanguageModel(unittest.TestCase):

    def setUp(self):
        self.x = ('MSKDDSIKKTVTIGLSAAIFFVLSCFASIPVGFNVSIETSVAFLAFIAVAFGPAVGFYVGLIGN'
                  'TIKDFILFGNVSWNWVLCSALIGFIYGLPHKIIDLKYQVFTKKKIVYFWLYQVAFNFIIWGFFA'
                  'PQSDLLIYGQPPKLVYLQSFLIVISNILAYSVVGIKLMSMYSCHYNKQATLIKINNS')
        self.y = ('MKKQDISVKTVVAIGIGAAVFVILGRFVVIPTGFPNTNIETSYAFLALISAIFGPFAGLMTGLV'
                  'GHAIKDFTTYGSAWWSWVICSGIIGCLYGWIGLKLNLSSGRFSRKSMIYFNIGQIIANIICWAL'
                  'IAPTLDILIYNEPANKVYTQGVISAVLNIISVGIIGTILLKAYASSQIKKGSLRKE')
        self.z = ('MEENIESVEEWVNKLDIETTKDIHVPKLLFDQVIGQDQAGEIVKKAALQRRHVILIGEPGTGKS'
                  'MLAQSMVDFLPKSELEDILVFPNPEDPNKPKIKTVPAGKGKEIVRQYQIKAEREKRDRSRSIMF'
                  'VIFSVVLLGIIAAIVLRSITLIFFAIMAAAFLYMAMAFNPVIRNERAMVPKLLVSHNPNDKPPF'
                  'VDSTGAHSGALLGDVRHDPFQSGGLETPAHERVEAGNIHKAHKGVLFIDEINLLRPEDQQAILT'
                  'ALQEKKYPISGQSERSAGAMVQTEPVPCDFVLVAAGNYDAIRNMHPALRSRIRGYGYEVVVNDY'
                  'MDDNDENRRKLVQFIAQEVEKDKKIPHFDKSAIIEVIKEAQKRSGRRNKLTLRLRELGGLVRVA'
                  'GDIAVSQKKTVVTAADVIAAKNLAKPLEQQIADRSIEIKKIYKTFRTEGSVVGMVNGLAVVGAD'
                  'TGMSEYTGVVLPIVAEVTPAEHKGAGNIIATGKLGDIAKEAVLNVSAVFKKLTGKDISNMDIHI'
                  'QFVGTYEGVEGDSASVSIATAVISAIENIPVDQSVAMTGSLSVRGDVLPVGGVTAKVEAAIEAG'
                  'LNKVIVPELNYSDIILDADHVNKIEIIPAKTIEDVLRVALVNSPEKEKLFDRISNLINAAKIIK'
                  'PQRPATPATTRAGNNAA')


class TestUnirep(TestLanguageModel):

    def test_extract(self):
        arch = 'unirep'
        cls, path = pretrained_language_models[arch]
        device = 'cuda'
        max_len = 1024
        input_dim = 1024
        embed_dim = 10
        self.align_fun = CCAaligner(input_dim, embed_dim, device=device)
        self.model = AlignmentModel(aligner=CCAaligner)
        self.model.load_language_model(cls, path, device=device)
        self.model.to(device)

        tokenizer = TAPETokenizer(vocab='unirep')
        res_x = torch.tensor([tokenizer.encode(self.x)]).to(device)
        # Assert that extract features are not nan
        res = self.model.lm(res_x)
        self.assertEqual(torch.isnan(res).sum().item(), 0)
        res_x = torch.tensor([tokenizer.encode(self.x)]).to(device)
        res = self.model.lm(res_x)
        self.assertEqual(torch.isnan(res).sum().item(), 0)
        res_x = torch.tensor([tokenizer.encode(self.x)]).to(device)
        res = self.model.lm(res_x)
        self.assertEqual(torch.isnan(res).sum().item(), 0)


class TestBert(TestLanguageModel):

    def test_extract(self):
        arch = 'bert'
        cls, path = pretrained_language_models[arch]
        device = 'cpu'
        max_len = 1024
        input_dim = 1024
        embed_dim = 10
        self.align_fun = CCAaligner(input_dim, embed_dim, device=device)
        self.model = AlignmentModel(aligner=CCAaligner)
        self.model.load_language_model(cls, path, device=device)
        self.model.to(device)

        tokenizer = TAPETokenizer(vocab='iupac')
        res_x = torch.tensor([tokenizer.encode(self.x)])
        res_y = torch.tensor([tokenizer.encode(self.y)])
        res_z = torch.tensor([tokenizer.encode(self.z)])
        # Assert that extract features are not nan
        res = self.model.lm(res_x)
        self.assertEqual(torch.isnan(res).sum().item(), 0)
        res = self.model.lm(res_y)
        self.assertEqual(torch.isnan(res).sum().item(), 0)
        res = self.model.lm(res_z)
        self.assertEqual(torch.isnan(res).sum().item(), 0)


class TestRoberta(TestLanguageModel):

    def test_extract(self):
        arch = 'roberta'
        cls, path = pretrained_language_models[arch]
        path = '/mnt/home/jmorton/ceph/checkpoints/pfam/checkpoint_gert'
        device = 'cpu'
        max_len = 1024
        input_dim = 1024
        embed_dim = 10
        self.align_fun = CCAaligner(input_dim, embed_dim, device=device)
        self.model = AlignmentModel(aligner=CCAaligner)
        self.model.load_language_model(cls, path, device=device)
        self.model.to(device)

        exp_x = self.model.lm.model.encode(' '.join(list(self.x)))
        exp_y = self.model.lm.model.encode(' '.join(list(self.y)))
        exp_z = self.model.lm.model.encode(' '.join(list(self.z)))

        # Assert that onehot encodings are equal
        res_x = seq2onehot(x)
        npt.assert_allclose(res_x.numpy(), exp_x.numpy())
        res_y = seq2onehot(y)
        npt.assert_allclose(res_y.numpy(), exp_y.numpy())
        res_z = seq2onehot(z)
        npt.assert_allclose(res_z.numpy(), exp_z.numpy())

        # Assert that extract features are not nan
        res = self.model.lm.model.extract_features(res_x)
        self.assertEqual(torch.isnan(res).sum().item(), 0)
        res = self.model.lm.model.extract_features(res_y)
        self.assertEqual(torch.isnan(res).sum().item(), 0)
        res = self.model.lm.model.extract_features(res_z)
        self.assertEqual(torch.isnan(res).sum().item(), 0)



if __name__ == '__main__':
    unittest.main()
