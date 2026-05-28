import torch

class SparseGraph:
    def __init__(self, num_hosts, k=5, device="cpu"):
        self.k = k
        self.device = device
        self.num_hosts = num_hosts
        self.neighbors = torch.full((num_hosts, k), -1, dtype=torch.long, device=device)
        self.ptr = torch.zeros(num_hosts, dtype=torch.long, device=device)
        self.risk = torch.zeros(num_hosts, device=device)
        
        # Mapa: hid (string) -> indeks (int)
        self.hid_to_idx = {}
        self.next_idx = 0

    def _get_idx(self, hid):
        if hid not in self.hid_to_idx:
            self.hid_to_idx[hid] = self.next_idx
            self.next_idx = (self.next_idx + 1) % self.num_hosts
        return self.hid_to_idx[hid]

    def update_interaction(self, hid_i, hid_j):
        i = self._get_idx(hid_i)
        j = self._get_idx(hid_j)
        
        idx = self.ptr[i]
        self.neighbors[i, idx] = j
        self.ptr[i] = (idx + 1) % self.k

    def update(self, decay=0.98):
        self.risk *= decay

    def get_risk(self, hid):
        # Upewnij się, że 'i' jest liczbą całkowitą z _get_idx
        i = self._get_idx(hid) 
        # Teraz używamy 'i' (indeksu), a nie 'hid' (stringa)
        mask = self.neighbors[i] != -1 
        if not mask.any(): 
            return torch.tensor(0.0, device=self.device)
        return self.risk[self.neighbors[i][mask]].mean()
    def is_quarantined(self, hid) -> bool:
        """
        Dron jest w kwarantannie gdy jego risk przekracza próg 0.85.
        Wywoływane przez ImmuneEngine przed każdym krokiem przetwarzania.
        """
        idx  = self._get_idx(hid)
        return bool(self.risk[idx].item() > 0.85)
