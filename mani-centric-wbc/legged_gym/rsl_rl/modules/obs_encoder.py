from abc import abstractmethod
from turtle import forward, setup

import torch
from git import List


class HistoryEncoder(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        history_len: int,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.net = self.setup_net(obs_dim, history_len)

    @abstractmethod
    def setup_net(self, obs_dim: int, history_len: int) -> torch.nn.Module:
        pass

    def forward(self, obs: torch.Tensor):
        if len(obs.shape) == 2 and obs.shape[1] == (self.history_len * self.obs_dim):
            obs = obs.view(-1, self.history_len, self.obs_dim)
        elif len(obs.shape) == 1 and obs.shape[0] == (self.history_len * self.obs_dim):
            obs = obs.view(1, self.history_len, self.obs_dim)
        assert obs.shape[1] == self.history_len
        assert obs.shape[2] == self.obs_dim
        return self.net(obs.permute(0, 2, 1))


class MLPHistoryEncoder(HistoryEncoder):
    def __init__(self, obs_dim: int, history_len: int, hidden_dims: List[int]):
        self.hidden_dims = hidden_dims
        super().__init__(obs_dim, history_len)

    def setup_net(self, obs_dim: int, history_len: int) -> torch.nn.Module:
        all_dims = [obs_dim * history_len] + self.hidden_dims
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(torch.nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor):
        return self.net(obs.view(-1, self.history_len * self.obs_dim))


class ConvHistoryEncoder(HistoryEncoder):
    def __init__(self, obs_dim: int, history_len: int, hidden_channels: List[int]):
        self.hidden_channels = hidden_channels
        if history_len == 10:
            assert hidden_channels[0] == 32
            assert hidden_channels[1] == 64
        else:
            raise NotImplementedError(f"history_len={history_len}")
        super().__init__(obs_dim, history_len)

    def setup_net(self, obs_dim: int, _: int) -> torch.nn.Module:
        all_dims = [obs_dim] + self.hidden_channels
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(
                torch.nn.Conv1d(
                    in_channels=all_dims[i],
                    out_channels=all_dims[i + 1],
                    kernel_size=4,
                    stride=2,
                )
            )
            if i < len(all_dims) - 2:
                layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Flatten())
        return torch.nn.Sequential(*layers)


class TransformerHistoryEncoder(HistoryEncoder):
    def __init__(
        self,
        obs_dim: int,
        history_len: int,
        hidden_dim: int,
        num_layers: int,
        dim_feedforward: int,
        n_head: int,
        use_positional_encoding: bool,
        output_latent_dim: int,
        concat_most_recent_obs: bool,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        super().__init__(obs_dim, history_len)
        self.in_proj = torch.nn.Linear(obs_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, output_latent_dim)
        self.concat_most_recent_obs = concat_most_recent_obs
        if use_positional_encoding:
            self.pos_enc = torch.nn.Parameter(torch.randn(history_len + 1, hidden_dim))
        else:
            self.pos_enc = None

    def setup_net(self, obs_dim: int, history_len: int) -> torch.nn.Module:
        return torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.n_head,
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
            ),
            num_layers=self.num_layers,
        )

    def forward(self, obs: torch.Tensor):
        if len(obs.shape) == 2 and obs.shape[1] == (self.history_len * self.obs_dim):
            obs = obs.view(-1, self.history_len, self.obs_dim)
        elif len(obs.shape) == 1 and obs.shape[0] == (self.history_len * self.obs_dim):
            obs = obs.view(1, self.history_len, self.obs_dim)
        assert obs.shape[1] == self.history_len
        assert obs.shape[2] == self.obs_dim
        # obs.shape = (B, T, C)
        obs_embed = self.in_proj(obs)
        if self.pos_enc is not None:
            obs_embed = obs_embed + self.pos_enc[: obs_embed.shape[1]]
        cls_token = torch.zeros(obs.shape[0], 1, self.hidden_dim, device=obs.device)
        embs = self.net(torch.cat([cls_token, obs_embed], dim=1))
        cls_emb = embs[:, 0]
        out_emb = self.out_proj(cls_emb)
        if self.concat_most_recent_obs:
            out_emb = torch.cat([out_emb, obs[:, -1]], dim=1)
        return out_emb
