import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
import numpy as np

class AutoSFScorer(RelationalScorer):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.A = self.get_option("A")
        self.K = self.get_option("K")


    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        s_emb_chunk = s_emb.chunk(self.K, dim=1)
        p_emb_chunk = p_emb.chunk(self.K, dim=1)
        o_emb_chunk = o_emb.chunk(self.K, dim=1)
        
        A_np = np.array(self.A)
        A_np_notzero = np.flatnonzero(A_np)

        print("autosf start")
        if combine == "spo":
            score = torch.zeros(n)
            for idx in A_np_notzero:
                ii = idx // self.K  # row: s
                jj = idx % self.K   # column: o
                A_ij = A_np[idx]
                # h^T g_K(A,r) t
                score += np.sign(A_ij) * (s_emb_chunk[ii] * p_emb_chunk[abs(A_ij)-1] * o_emb_chunk[jj]).sum(dim=1)

        elif combine == "sp_":
            m = o_emb.size(0)
            score = torch.zeros(n,m).to('cuda:0')
            for idx in A_np_notzero:
                ii = idx // self.K  # row: s
                jj = idx % self.K   # column: o
                A_ij = A_np[idx]
                # h^T g_K(A,r) t
                score += np.sign(A_ij) * ((s_emb_chunk[ii] * p_emb_chunk[abs(A_ij)-1]).mm(o_emb_chunk[jj].transpose(0,1)))

        elif combine == "_po":
            m = s_emb.size(0)
            score = torch.zeros(n,m).to(self.config.get("job.device"))
            for idx in A_np_notzero:
                ii = idx // self.K  # row: s
                jj = idx % self.K   # column: o
                A_ij = A_np[idx]
                # h^T g_K(A,r) t
                score += np.sign(A_ij) * ((o_emb_chunk[jj] * p_emb_chunk[abs(A_ij)-1]).mm(s_emb_chunk[jj].transpose(0,1)))

        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return score.view(n, -1)      



class AutoSF(KgeModel):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=AutoSFScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )





