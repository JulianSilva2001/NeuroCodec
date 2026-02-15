import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dac
import math
from mamba_ssm import Mamba

# Import utilities
from utility.layers import GraphConvolution
from utility.utils import normalize_A, generate_cheby_adj
from utility.utils import ChannelwiseLayerNorm, ResBlock, Conv1D


# Import existing modules
# We redefine EEGEncoder to avoid hardcoded shapes in M3ANET.py
from models.groupmamba import GroupMamba 

class Chebynet(nn.Module):
    def __init__(self, in_channel=128, k_adj=3):
        super(Chebynet, self).__init__()
        self.K = k_adj
        self.gc = nn.ModuleList()
        for i in range(k_adj):
            self.gc.append(GraphConvolution(in_channel, in_channel))

    def forward(self, x ,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result

class FlexibleEEGEncoder(nn.Module):
    def __init__(self, num_electrodes=128, k_adj=3, enc_channel=128, feature_channel=64, kernel_size=8,
                 norm='ln', K=160, kernel=3, stride=1): # Changed stride 4 -> 1
        super().__init__()
        self.stride = stride
        self.K = k_adj
        
        # BN over Electrodes (Channels)
        self.BN1 = nn.BatchNorm1d(num_electrodes)
        
        self.layer1 = Chebynet(num_electrodes, k_adj)
        # Stride=1 to preserve time resolution
        self.projection = nn.Conv1d(num_electrodes, feature_channel, kernel_size, bias=False, stride=self.stride)
        
        # Learnable Adjacency Matrix
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes , num_electrodes))
        nn.init.xavier_normal_(self.A)

        # Removed Pooling from ResBlocks?
        # Standard ResBlock has MaxPool1d(3) inside.
        # We need a ResBlock WITHOUT pooling if we want to keep resolution.
        # Or we redefine the encoder sequence to use dilated convolutions instead?
        # Easiest: Use Conv1D with stride=1 instead of ResBlock if ResBlock forces pooling.
        # Let's check ResBlock definition in utils.py again.
        # It has self.maxpool = nn.MaxPool1d(3).
        # We must redefine ResBlock or use a custom block here.
        
        # Let's use simple Conv1D blocks to keep it clean and adaptable
        self.dropout = nn.Dropout(0.3)
        self.eeg_encoder = nn.Sequential(
            ChannelwiseLayerNorm(feature_channel),
            Conv1D(feature_channel, feature_channel, 1),
            # ResBlock(feature_channel, feature_channel), # Removed due to pooling
            Conv1D(feature_channel, feature_channel, 3, padding=1),
            nn.PReLU(),
            self.dropout,
            # ResBlock(feature_channel, enc_channel),
            Conv1D(feature_channel, enc_channel, 3, padding=1),
            nn.PReLU(),
            self.dropout,
            # ResBlock(enc_channel, enc_channel),
            Conv1D(enc_channel, enc_channel, 3, padding=1),
            nn.PReLU(),
            self.dropout,
            Conv1D(enc_channel, feature_channel, 1),
        )

    def forward(self, spike):
        # spike: (B, 128, T)
        
        # Normalize Electrodes
        # BN expects (B, C, L) -> (B, 128, T)
        spike = self.BN1(spike)
        
        # Chebynet (GCN)
        # Expects (B, 128, T) ?
        # normalize_A expects A on same device
        if self.A.device != spike.device:
            self.A = self.A.to(spike.device)
            
        L = normalize_A(self.A)
        
        # Chebynet implementation in M3ANET seems to take (x, adj)
        # GraphConvolution usually expects (B, N, T) or (B, N, C)?
        # M3ANET GraphConvolution: x is (B, N, T)?
        # Let's assume it works on (B, N, T) and propagates info across N.
        output = self.layer1(spike, L)
        
        # Projection (Conv1d)
        # (B, 128, T) -> (B, feature_channel, T')
        output = self.projection(output)    
        output = self.eeg_encoder(output)   

        return output



class NeuroCodecBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads=8, d_state=16, d_conv=4, expand=2, dropout=0.1, backbone='mamba'):
        super().__init__()
        self.backbone_type = backbone
        
        # 1. Cross Attention
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Backbone Core (Mamba or Transformer)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        if backbone == 'mamba':
            self.core = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        elif backbone == 'transformer':
            # Causal Self-Attention Layer
            # We use TransformerEncoderLayer but must ensure causal masking in forward()
            self.core = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=n_heads, 
                dim_feedforward=hidden_dim*4, 
                dropout=dropout, 
                activation='gelu',
                batch_first=True,
                norm_first=True # Pre-Norm like Mamba/GPT
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, eeg_k, eeg_v):
        # x: (B, T, D) - Audio (Query)
        # eeg_k, eeg_v: (B, S, D) - EEG (Key/Value)
        
        # A. Cross Attention with Residual
        x_norm = self.ln1(x)
        attn_out, attn_weights = self.attn(query=x_norm, key=eeg_k, value=eeg_v)
        x = x + self.dropout1(attn_out)
        
        # B. Backbone with Residual
        if self.backbone_type == 'mamba':
            x_norm = self.ln2(x)
            core_out = self.core(x_norm)
            x = x + self.dropout2(core_out)
            
        elif self.backbone_type == 'transformer':
            # Manually handle residual because TransformerEncoderLayer includes it?
            # Standard nn.TransformerEncoderLayer(x) does: x + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x + self.dropout(self.self_attn(self.norm1(x))...))))))
            # Wait, if norm_first=True, it expects input x.
            # We need to provide CURRENT sequence mask for causality.
            
            # Generate Causal Mask
            B, T, D = x.shape
            # mask: (T, T) - -inf above diagonal
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            
            # Forward
            # Note: TransformerEncoderLayer applies its own residual/norm. 
            # If we defined it with norm_first=True, it effectively does Pre-Norm.
            # But here we have specific structure: LN -> Core -> Add.
            # Let's wrap it to match Mamba block structure if possible, or just use it as is.
            # If we use nn.TransformerEncoderLayer, it IS a full block (Attn + FFN + Residuals).
            # Mamba block is: Norm -> Mamba -> Add.
            
            # Let's just pass x to it, it handles residuals.
            # But wait, Mamba is just the mixer. TransformerEncoderLayer is Mixer+MLP+Residuals.
            # This is a slight architectural difference. Mamba also has internal gates/projections.
            # It is comparable.
            
            core_out = self.core(x, src_mask=causal_mask, is_causal=True)
            x = core_out # It includes residual connection inside
            
        return x, attn_weights
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return x

class NeuroCodec(nn.Module):
    def __init__(self, 
                 dac_model_type='44khz',
                 eeg_in_channels=128,
                 hidden_dim=256,
                 num_layers=4,
                 backbone='mamba'):
        super().__init__()
        
        # 1. Frozen DAC Backbone
        print(f"Loading Frozen DAC ({dac_model_type})...")
        model_path = dac.utils.download(model_type=dac_model_type)
        self.dac = dac.DAC.load(model_path)
        
        # Freeze DAC
        for param in self.dac.parameters():
            param.requires_grad = False
        self.dac.eval()
        print("DAC Loaded and Frozen.")
        
        # 2. EEG Encoder (Keeping GCN/EEGEncoder from M3ANet)
        self.eeg_encoder = FlexibleEEGEncoder(num_electrodes=eeg_in_channels, 
                                      enc_channel=64, 
                                      feature_channel=64, 
                                      norm='ln')

        # 4. Input Projections
        # We project z_mix (1024) -> hidden_dim
        self.audio_proj = nn.Linear(1024, hidden_dim)
        
        # 5. Stacked Layers (Fusion + Generator)
        self.layers = nn.ModuleList([
            NeuroCodecBlock(hidden_dim=hidden_dim, dropout=0.3, backbone=backbone) # Increased dropout to 0.3
            for _ in range(num_layers)
        ])
        
        # Projections
        self.eeg_proj = nn.Linear(64, hidden_dim)       # EEG dim -> hidden
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=5000)
        
        # 6. Output Head (Regression to Z)
        # Hidden -> 1024 (DAC Latent Dim)
        self.output_proj = nn.Linear(hidden_dim, 1024)
        
        # 7. Envelope Projection Head (New for Step 13)
        # EEG Features (64) -> Envelope (1)
        self.envelope_proj = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, mixture, eeg):
        """
        mixture: (B, 1, T_audio) [Time Domain]
        eeg: (B, 128, T_eeg)     [Time Domain]
        """
        
        # 1. Encode Audio with Frozen DAC
        with torch.no_grad():
            z_mix, codes_mix, _, _, _ = self.dac.encode(mixture)
            
        # 2. Encode EEG
        eeg_feat = self.eeg_encoder(eeg)
        
        # 3. Envelope Prediction (Auxiliary Task)
        # eeg_feat: (B, 64, T_eeg) -> (B, 1, T_eeg) -> (B, T_eeg)
        envelope_pred = self.envelope_proj(eeg_feat).squeeze(1)
        
        # 4. Preparation for Core
        # z_mix: (B, 1024, T)
        # Transpose to (B, T, 1024) for Linear
        x_audio = z_mix.transpose(1, 2)
        x_audio = self.audio_proj(x_audio) # (B, T, H)
        
        # B. Project EEG
        # (B, 64, T_eeg) -> (B, T_eeg, 64) -> (B, T_eeg, H)
        x_eeg = self.eeg_proj(eeg_feat.transpose(1, 2)) 
        
        # Apply Positional Encoding
        # Standard scaling to balance Embedding variance with PE variance
        scale = math.sqrt(self.audio_proj.out_features) # hidden_dim
        x_audio = self.pos_encoder(x_audio * scale)
        x_eeg = self.pos_encoder(x_eeg * scale)
        
        # (Fixing indentation/logic here if PE scaling was debated, keeping it simple)
        # x_audio = self.pos_encoder(x_audio)
        # x_eeg = self.pos_encoder(x_eeg)
        
        # 4. Stacked Layers
        x = x_audio
        last_attn_weights = None
        
        for layer in self.layers:
            x, last_attn_weights = layer(x, x_eeg, x_eeg)
        
        x_hidden = x
        
        # 5. Output Head (Regression)
        # (B, T, H) -> (B, T, 1024)
        z_pred = self.output_proj(x_hidden)
        
        # Transpose back to (B, 1024, T) for loss/DAC
        z_pred = z_pred.transpose(1, 2)
        
        return z_pred, codes_mix, z_mix, eeg_feat, last_attn_weights, envelope_pred

if __name__ == "__main__":
    # Internal Verification
    model = NeuroCodec()
    print("NeuroCodec Instantiated.")