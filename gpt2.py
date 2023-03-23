import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from utils import FullyConnected, LayerNorm
from multi_head_attention import MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPT2Block(nn.Module):
    """
    The gpt2 decoder layer is composed of a multi-head self-attention mechanism,
    followed by a simple, position-wise fully connected feed-forward network. 
    This architecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, num_heads, d_model, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(GPT2Block, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      dropout_rate=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=d_model,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm2 = LayerNorm(d_model, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # apply layer normalization (modified from original transformer decoder)
        x = self.layernorm1(x)
        
        # calculate Self-Attention using Multi-Head Attention
        mha_output = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, d_model)

        # skip connection
        # apply layer normalization on sum of the input and the attention output to get the output of the multi-head attention layer
        skip_x_attention = self.layernorm2(x + mha_output)

        # pass the output of the multi-head attention layer through a ffn
        ffn_output = self.ffn(skip_x_attention)

        # apply dropout layer to ffn output during training
        ffn_output = self.dropout_ffn(ffn_output)

        # sum of the output from multi-head attention (skip connection) and ffn output to get the output of the encoder layer
        encoder_layer_out = skip_x_attention + ffn_output

        return encoder_layer_out

class GPT2LanguageModel(nn.Module):

    def __init__(self, num_layers, num_heads, d_model, fully_connected_dim,
                input_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(GPT2LanguageModel, self).__init__()
        
        self.d_model = d_model
        self.maximum_position_encoding = maximum_position_encoding

        self.token_embedding = nn.Embedding(input_vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, maximum_position_encoding, d_model))
        
        self.layers = nn.ModuleList([GPT2Block(num_heads, d_model, fully_connected_dim, 
                                               dropout_rate=dropout_rate, layernorm_eps=layernorm_eps) 
                                     for _ in range(num_layers)])
        
        self.linearnorm_fc = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, input_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def padding_mask(self, x):
        mask = torch.tril(torch.ones((x.shape[-1], x.shape[-1])))
        mask = mask.unsqueeze(0).unsqueeze(1)
        
        return mask.to(device)

    def forward(self, x):
        padding_mask = self.padding_mask(x)

        seq_len = x.shape[-1]
        
        token_embeddings = self.token_embedding(x)

        x = self.dropout(token_embeddings + self.position_embedding[:, :seq_len, :])
        
        for layer in self.layers:
            x = layer(x, padding_mask)
        
        x = self.linearnorm_fc(x)
        logits = self.fc(x)

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.maximum_position_encoding:]
            
            # get the predictions
            logits = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

if __name__ == '__main__':
    # DEFAULT GPT-2 base PARAMETERS
    batch_size = 512
    vocab_size = 50257
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_head = 12
    
    intermediate_size = 4 * hidden_size
    dropout = 0.1
    max_positional_emnddings = 1024
    layer_norm_eps = 1e-12


    a = torch.randint(1, 100, (3, 300))
    
    model = GPT2LanguageModel(
                    num_layers=num_hidden_layers,
                    num_heads=num_attention_head,
                    d_model=hidden_size,
                    fully_connected_dim=intermediate_size,
                    input_vocab_size=vocab_size,
                    maximum_position_encoding=max_positional_emnddings,
                    dropout_rate=dropout,
                    layernorm_eps=layer_norm_eps
                )

    start = time.time()

    for i in range(2):
        y = model(a)

    print(f'INFERENCE TIME = {time.time() - start}sec')
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'This implementation of GPT-2 has {round(x / 1e6)}M parameters.')
    
