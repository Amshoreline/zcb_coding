import math
import torch


def embedding(pos_list, d_model, base=10_000):
    '''
    Params:
        pos_list: torch.Tensor(k, )
        d_model: int
        base: int
    Return:
        pos_embed: torch.Tensor(k, d_model)
    '''
    assert d_model % 2 == 0
    half_d = torch.arange(d_model // 2).float()
    half_d = base ** (2 * half_d / d_model)  # (d_model // 2, )
    #
    pos_list = (pos_list * 2 * math.pi)[:, None]  # (k, 1)
    pos_list = pos_list / half_d[None]  # (k, d_model // 2)
    pos_embed = torch.cat(
        [pos_list.sin()[..., None], pos_list.cos()[..., None]],
        dim=2
    )  # (k, d_model // 2, 2)
    pos_embed = pos_embed.view(-1, d_model)
    return pos_embed


def rotated_pe(feats, base=10_000):
    '''
    Params:
        feats: torch.Tensor(bs, seq_len, d_model)
        base: int
    Return:
        pos_embed: torch.Tensor(k, d_model)
    '''
    bs, seq_len, d_model = feats.shape
    assert d_model % 2 == 0
    thetas = base ** (-2 * torch.arange(d_model // 2).float() / d_model)  # (d_model // 2, )
    pos_list = torch.arange(seq_len).float()  # (seq_len, )
    full_thetas = pos_list.view(-1, 1) * thetas.view(1, -1)  # (seq_len, d_model // 2)
    full_thetas = full_thetas[..., None]  # (seq_len, d_model // 2, 1)
    full_theta_cnums = torch.cat([full_thetas.cos(), full_thetas.sin()], dim=-1)  # (seq_len, d_model // 2, 2)
    full_theta_cnums = torch.view_as_complex(full_theta_cnums)
    #
    feats_cnums = torch.view_as_complex(feats.view(bs, seq_len, d_model // 2, 2))
    feats_embed = feats_cnums * full_theta_cnums[None]
    feats_embed = torch.view_as_real(feats_embed)  # (bs, seq_len, d_model // 2, 2)
    feats_embed = feats_embed.view(bs, seq_len, -1)
    return feats_embed


def main():
    print(embedding(torch.tensor([0, 1, 2, 3, 4, 5]), 2))    
    feats = torch.randn(3, 100, 4) * 100
    print(feats[0, 50])
    print(rotated_pe(feats)[0, 50])


if __name__ == '__main__':
    main()
