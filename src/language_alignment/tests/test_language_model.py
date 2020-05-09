import torch
import unittest
from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import CCAaligner


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

        x = self.model.lm.model.encode(' '.join(list(x)))
        y = self.model.lm.model.encode(' '.join(list(y)))
        z = self.model.lm.model.encode(' '.join(list(z)))

        res = self.model.lm.model.extract_features(x)
        assert torch.isnan(res).sum().item() > 0

        res = self.model.lm.model.extract_features(y)
        assert torch.isnan(res).sum().item() > 0

        res = self.model.lm.model.extract_features(z)
        assert torch.isnan(res).sum().item() > 0


if __name__ == '__main__':
    unittest.main()
