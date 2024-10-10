import numpy as np
import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, d_model, num_heads, use_future_mask=False):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_v = d_model // num_heads
        self.sqrt_d_v = np.sqrt(self.d_v)
        #
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        #
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        #
        self.use_future_mask = use_future_mask
    
    def forward(self, x):
        '''
        Params:
            x: torch.Tensor(bs, seq_len, d_model)
        Return:
            out: torch.Tensor(bs, seq_len, d_model)
        '''
        bs, seq_len, _ = x.size()
        #
        q = self.q_linear(x)  # (bs, seq_len, d_model)
        k = self.k_linear(x)
        v = self.v_linear(x)
        # multihead
        q = q.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)  # (bs, num_heads, seq_len, d_v)
        k = k.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        # attention map
        attn = q @ k.transpose(-1, -2)  # (bs, num_heads, seq_len, seq_len)
        attn = attn / self.sqrt_d_v
        if self.use_future_mask:
            future_mask = torch.arange(seq_len).view(-1, 1) < torch.arange(seq_len).view(1, -1)
            attn[..., future_mask] = -100
        attn = attn.softmax(dim=-1)
        # print((attn[0, 0] * 100).round())
        # attention
        out = attn @ v  # (bs, num_heads, seq_len, d_v)
        out = out.transpose(1, 2).contiguous().view(bs, seq_len, -1)  # (bs, seq_len, d_model)
        return self.out_linear(out)


class LoRALinear(nn.Linear):

    def __init__(self, d_model, d_r):
        super().__init__(d_model, d_model, bias=False)
        self.weight.requires_grad = False
        #
        self.lora_A = nn.Linear(d_model, d_r, bias=False)
        self.lora_B = nn.Linear(d_r, d_model, bias=False)
        nn.init.kaiming_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        #
        self.merged = False
    
    def train(self, mode=False):
        super().train(mode)
        if mode:
            if self.merged:
                self.weight.data -= (self.lora_A.weight.T @ self.lora_B.weight.T).T
                self.merged = False
        else:
            if not self.merged:
                self.weight.data += (self.lora_A.weight.T @ self.lora_B.weight.T).T
                self.merged = True

    def forward(self, x):
        '''
        Params:
            x: torch.Tensor(bs, ..., d_model)
        Output:
            x: torch.Tensor(bs, ..., d_model)
        '''
        if self.merged:
            return x @ self.weight.T
        else:
            ori_out = x @ self.weight.T
            new_out = x @ self.lora_A.weight.T @ self.lora_B.weight.T
            return ori_out + new_out


def main():
    # attn_layer = Attention(4, 2, use_future_mask=True)
    # x = torch.randn(2, 5, 4)
    # out = attn_layer(x)
    # print(out.shape)
    #
    # ori_linear = nn.Linear(4, 4, bias=False)
    # lora_linear = LoRALinear(4, 2)
    # lora_linear.load_state_dict(ori_linear.state_dict(), strict=False)
    # x = torch.randn(2, 1, 4) * 10
    # print((ori_linear(x) - lora_linear(x)).abs().max())
    # lora_linear.eval()
    # print((ori_linear(x) - lora_linear(x)).abs().max())
    # lora_linear.train()
    # print((ori_linear(x) - lora_linear(x)).abs().max())
    #
    attn_layer = Attention(4, 2)
    q_state_dict = attn_layer.q_linear.state_dict()
    attn_layer.q_linear = LoRALinear(4, 2)
    attn_layer.q_linear.load_state_dict(q_state_dict, strict=False)
    v_state_dict = attn_layer.v_linear.state_dict()
    attn_layer.v_linear = LoRALinear(4, 2)
    attn_layer.v_linear.load_state_dict(v_state_dict, strict=False)
    trainable_params = []
    for name, param in attn_layer.named_parameters():
        if 'lora_' in name:
            print('train', name, param.size())
            trainable_params.append(param)
        else:
            param.requires_grad = False
    optimizer = torch.optim.Adam(trainable_params, lr=1e-5)
    x = torch.ones(2, 5, 4)
    for _ in range(1000):
        out = attn_layer(x)
        loss = torch.sum(out)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()