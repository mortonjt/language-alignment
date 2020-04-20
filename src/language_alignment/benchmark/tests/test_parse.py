import unittest
from language_alignment.util import get_data_path
from language_alignment.benchmark.parse import msa2fasta


class TestParse(unittest.TestCase):

    def setUp(self):
        self.malisam = get_data_path('d1uoua1d1fs1b1.manual.ali',
                                     subfolder='malisam')

    def test_parse(self):
        seqs = msa2fasta(self.malisam)
        exp_seqs = [
            ('SAQGAQIGAMLMAIRLRGMDLEETSVLTQALAQSGQQLEWQ'
             'SEQLVPADGILYAARDVTATVDSLPLITASILSKKLVEGL'),
            ('DPVPLPNVNAAILKKVIQWCTHHKDDPVWDQEFLKVDQGTL'
             'FELILAANYLDIKGLLDVTCKTVANMIKGKTPEEIRKTFN')
        ]
        self.assertListEqual(seqs, exp_seqs)


if __name__ == "__main__":
    unittest.main()
