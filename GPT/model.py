import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
import time

delay = 0

# 模型结构

class SingleHeadAttention(nn.Module):
    # 单头注意力机制
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)

        # 尝试学习新的写法，attention_mask 通过 register_buffer 注册
        # 因为不用计算 梯度，所以节约内存和显存，速度也更快
        self.register_buffer(
            'attention_mask', 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)   # @ 就是 torch.matmul 的简化写法
        # 一定要在 softmax 前除以 sqrt(head_size)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0, 
            float('-inf')
        ) / math.sqrt(hidden_size)  # 这里的 hidden_size 其实是 head_size，因为是单头
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads], 
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    # 实际上就是 MLP
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)


# 接下来就是一个完整的 Block

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# 以后会讲  MLA ,  MOE, DPO 完全手写
# 完整的  GPT model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.block_size = config.block_size
        # linear (4 -> 8)； weight shape 是记上是 8 * 4，
        # 所以 embedding weight 和 lm_head weight 是共享的
        # 这里学习一下 tie weight。
        # 这是为了减少参数，加快训练；（现在 25的 SLM 很多都这样做了，注意⚠️）
        # self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # 改进点：
        # postion embedding 从0, 1, xxx embdding 升级到 rope
        # norm layer norm -> rms norm
        # mlp -> swiglu
        # mha -> gpa


        #
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这里使用的是正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 是输入的 token ids
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)

        # seq 长度是这次输入的最大长度
        pos_emb = self.position_embedding_table(
            # 要确保 位置编码和输入的 idx 在同一个设备上
            torch.arange(seq_len, device=idx.device)
        )
        # 有一个经典题目：为什么 embedding 和 position 可以相加？
        x = token_emb + pos_emb   # shape is (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)   # shape is (batch, seq_len, vocab_size)
        
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=5000):
        assert idx.size(0) == 1, "please input one sentence"
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 block_size 个token
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 获取预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            if idx[0, -1] == self.eos_token:
                break
            generated_word = self.id2string(idx[:, -1])  # 获取刚生成的 token 对应的字符
            print(generated_word, end="", flush=True)  # 打印当前字符，并确保立即刷新显示
            time.sleep(delay)  # 延迟，用于控制打印速度
        print()
        return idx
        # return self.enc.decode(idx.squeeze().detach().cpu().numpy().tolist())

    def string2id(self, s: str):
        # 将输入字符串转换为 token ID 列表
        ids = self.enc.encode(s)
        # 添加结束符
        ids.append(self.eos_token)
        return torch.tensor([ids])

    def id2string(self, idx):
        # 将张量转换为 CPU 并解压缩（如果必要）
        idx = idx.squeeze().cpu().numpy()

        # 如果 idx 是标量，转换为列表
        if idx.ndim == 0:  # 检查是否是标量
            idx = [idx.item()]  # 使用 .item() 获取标量值并放入列表
        else:
            idx = idx.tolist()  # 如果是数组，直接转换为列表

        # 调用解码器
        return self.enc.decode(idx)

    def query(self, s:str, max_new_tokens=5000):
        idx = self.string2id(s)
        device = next(self.parameters()).device
        # 将输入字符串转换为 token ID，并将其移动到与模型相同的设备
        idx = self.string2id(s).to(device)
        idx = self.generate(idx, max_new_tokens)
        return self.id2string(idx)
        