import torch
import unittest
from language_alignment.dataset.dataset import seq2onehot
from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import CCAaligner
import numpy.testing as npt


class TestRobertaExtractFeatures(unittest.TestCase):

    def setUp(self):
        arch = 'roberta'
        cls, path = pretrained_language_models[arch]
        path = '/mnt/home/jmorton/ceph/checkpoints/pfam/checkpoint_gert'
        device = 'cpu'
        max_len = 1024
        input_dim = 1024
        embed_dim = 10
        self.align_fun = CCAaligner(input_dim, embed_dim, max_len, device=device)
        self.model = AlignmentModel(aligner=CCAaligner)
        self.model.load_language_model(cls, path, device=device)
        self.model.to(device)

    def test_extract(self):
        x = ('MSKDDSIKKTVTIGLSAAIFFVLSCFASIPVGFNVSIETSVAFLAFIAVAFGPAVGFYVGLIGN'
             'TIKDFILFGNVSWNWVLCSALIGFIYGLPHKIIDLKYQVFTKKKIVYFWLYQVAFNFIIWGFFA'
             'PQSDLLIYGQPPKLVYLQSFLIVISNILAYSVVGIKLMSMYSCHYNKQATLIKINNS')
        y = ('MKKQDISVKTVVAIGIGAAVFVILGRFVVIPTGFPNTNIETSYAFLALISAIFGPFAGLMTGLV'
             'GHAIKDFTTYGSAWWSWVICSGIIGCLYGWIGLKLNLSSGRFSRKSMIYFNIGQIIANIICWAL'
             'IAPTLDILIYNEPANKVYTQGVISAVLNIISVGIIGTILLKAYASSQIKKGSLRKE')
        z = ('MEENIESVEEWVNKLDIETTKDIHVPKLLFDQVIGQDQAGEIVKKAALQRRHVILIGEPGTGKS'
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

        exp_x = self.model.lm.model.encode(' '.join(list(x)))
        exp_y = self.model.lm.model.encode(' '.join(list(y)))
        exp_z = self.model.lm.model.encode(' '.join(list(z)))

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
