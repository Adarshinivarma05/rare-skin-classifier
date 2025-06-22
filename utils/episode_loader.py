import random, torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class EpisodeDataset(Dataset):
    """
    Generates N-way K-shot episodes on-the-fly.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Any dataset that returns (image, label).
    n_way : int
        Classes per episode.
    k_shot : int
        Support images per class.
    q_queries : int
        Query  images per class.
    episodes : int
        How many episodes constitute one epoch (== len(self)).
    """
    def __init__(self, dataset, n_way=3, k_shot=1, q_queries=1, episodes=1000):
        self.dataset  = dataset
        self.n_way    = n_way
        self.k_shot   = k_shot
        self.q_queries= q_queries
        self.episodes = episodes
        self._index_by_class()

    # ------------------------------------------------------------------
    def _index_by_class(self):
        self.class_to_indices = defaultdict(list)
        for idx, (_, y) in enumerate(self.dataset):
            self.class_to_indices[int(y)].append(idx)
        self.classes = list(self.class_to_indices.keys())

    # ------------------------------------------------------------------
    def __len__(self):
        return self.episodes

    # ------------------------------------------------------------------
    def __getitem__(self, _):
        # 1) pick N distinct classes
        chosen_cls = random.sample(self.classes, self.n_way)

        support_imgs, support_lbls, query_imgs, query_lbls = [], [], [], []

        for new_id, cls in enumerate(chosen_cls):
            pool = random.sample(self.class_to_indices[cls],
                                 self.k_shot + self.q_queries)
            sup_idx, qry_idx = pool[:self.k_shot], pool[self.k_shot:]

            support_imgs.extend([self.dataset[i][0] for i in sup_idx])
            support_lbls.extend([new_id] * self.k_shot)

            query_imgs.extend([self.dataset[i][0] for i in qry_idx])
            query_lbls.extend([new_id] * self.q_queries)

        return (torch.stack(support_imgs),
                torch.tensor(support_lbls),
                torch.stack(query_imgs),
                torch.tensor(query_lbls))


def get_episode_loader(dataset, n_way, k_shot, q_queries,
                       episodes_per_epoch, batch_size=1, shuffle=True):
    epis_ds = EpisodeDataset(dataset, n_way, k_shot, q_queries, episodes_per_epoch)
    return DataLoader(epis_ds, batch_size=batch_size, shuffle=shuffle)
