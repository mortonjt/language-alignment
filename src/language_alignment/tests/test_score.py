import unittest
import numpy as np
from language_alignment.score import domain_score, score_alignment
import pandas as pd
import pandas.util.testing as pdt


class TestScoreAlignment(unittest.TestCase):
    def test_score_alignment(self):
        pred_edges = pd.DataFrame(
            [(1, 2),
             (2, 3),
             (3, 4),
             (4, 5)],
            columns=['source', 'target']
        )
        truth_edges = pd.DataFrame(
            [(2, 3),
             (3, 4),
             (4, 5),
             (5, 6)],
            columns=['source', 'target']
        )
        score_alignment(pred_edges, truth_edges, total_length=4)

class TestDomain(unittest.TestCase):

    def setUp(self):
        self.sx = ('MVTIRADEISNIIRERIEQYNREVKIVNTGTVLQVGDGIARIHGLDEVMAGELVE'
                   'FEEGTIGIALNLESNNVGVVLMGDGLMIQEGSSVKATGRIAQIPVSEAYLGRVIN'
                   'ALAKPIDGRGEISASESRLIESPAPGIISRRSVYEPLQTGLIAIDSMIPIGRGQR'
                   'ELIIGDRQTGKTAVATDTILNQQGQNVICVYVAIGQKASSVAQVVTTFQERGAME'
                   'YTIVVAETADSPATLQYLAPYTGAALAEYFMYRERHTSIIYDDPSKQAQAYRQMS'
                   'LLLRRPPGREAYPGDVFYLHSRLLERAAKSSSNLGEGSMTALPIVETQSGDVSAY'
                   'IPTNVISITDGQIFLSADLFNAGIRPAINVGISVSRVGSAAQIKAMKQVAGKLKL'
                   'ELAQFAELEAFAQFASDLDKATQNQLARGQRLRELLKQSQAAPLAVEEQIMTIYT'
                   'GTTGYLDSLEIGQVRKFLVALRAYVKTNKPQFQEIISSTKTFTEEAESLLKEAIQ'
                   'EQMDRFLLQEQA')
        self.sy = ('MKTGKIIKVSGPLVVAEGMDEANVYDVVKVGEKGLIGEIIEMRGDKASIQVYEET'
                   'SGIGPGDPVITTGEPLSVELGPGLIESMFDGIQRPLDAFMKAANSAFLSKGVEVK'
                   'SLNREKKWPFVPTAKVGDKVSAGDVIGTVQETAVVLHRIMVPFGVEGTIKEIKAG'
                   'DFNVEEVIAVVETEKGDKNLTLMQKWPVRKGRPYARKLNPVEPMTTGQRVIDTFF'
                   'PVAKGGAAAVPGPFGAGKTVVQHQVAKWGDTEIVVYVGCGERGNEMTDVLNEFPE'
                   'LKDPKTGESLMKRTVLIANTSNMPVAAREASIYTGITIAEYFRDMGYSVSIMADS'
                   'TSRWAEALREMSGRLEEMPGDEGYPAYLGSRLADYYERAGKVVALGKDGREGAVT'
                   'AIGAVSPPGGDISEPVTQSTLRIVKVFWGLDAQLAYKRHFPSINWLTSYSLYLEK'
                   'MGEWMDAHVADDWSALRTEAMALLQEEANLEEIVRLVGMDALSEGDRLKLEVAKS'
                   'IREDYLQQNAFHENDTYTSLNKQYKMLNLILSFKHEAEKALEAGVYLDKVLKLPV'
                   'RDRIARSKYISEEEISKMDDILVELKSEMNKLISEGGVLNA')

        cols = ['protein', 'domain', 'source', 'domain_id', 'start', 'end', 'length']
        rows = [411563, 411564, 411565]
        data = [
            ['A0A320', 'PF00006', 'ATP-synt_ab', 'Pfam-A', 150, 365, 215],
            ['A0A320', 'PF02874', 'ATP-synt_ab_N', 'Pfam-A', 26, 93, 67],
            ['A0A320', 'PF00306', 'ATP-synt_ab_C', 'Pfam-A', 372, 497, 125]
        ]
        self.dom_x = pd.DataFrame(data, index=rows, columns=cols)
        rows = [246300, 246301, 246302]
        data = [
            ['Q0TPW7', 'PF00006', 'ATP-synt_ab', 'Pfam-A', 212, 435, 223],
            ['Q0TPW7', 'PF02874', 'ATP-synt_ab_N', 'Pfam-A', 6, 68, 62],
            ['Q0TPW7', 'PF16886', 'ATP-synt_ab_Xtn', 'Pfam-A', 84, 203, 119]
        ]
        self.dom_y = pd.DataFrame(data, index=rows, columns=cols)
        self.edges = pd.read_csv('edges.csv')

    def test_domain_score(self):
        cols = ['tp', 'fp', 'len']
        index = ['PF00006', 'PF02874']

        data = [
            [64, 30, 214],
            [23, 13, 66]
        ]

        exp = pd.DataFrame(data, columns=cols, index=index)
        res = domain_score(self.edges, self.sx, self.dom_x,
                           self.sy, self.dom_y)
        pdt.assert_frame_equal(exp, res)


if __name__ == "__main__":
    unittest.main()
