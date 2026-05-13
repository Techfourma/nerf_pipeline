import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_freqs):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.output_dim = input_dim * (2 * num_freqs)
        self.register_buffer('freq_bands', 2.0 ** torch.arange(num_freqs))

    def forward(self, x):
        encoded = x[..., None] * self.freq_bands
        encoded = torch.stack([torch.sin(encoded), torch.cos(encoded)], dim=-1)
        return encoded.reshape(x.shape[:-1] + (-1,))

class NeRFModel(nn.Module):
    def __init__(self, depth=8, width=256, input_ch=3, input_ch_views=3,
                 num_freqs=10, num_freqs_views=4, skip_layer=4):
        super().__init__()
        self.depth = depth
        self.width = width
        self.skip_layer = skip_layer

        self.pos_encoder = PositionalEncoding(input_ch, num_freqs)
        self.view_encoder = PositionalEncoding(input_ch_views, num_freqs_views)

        input_ch_encoded = input_ch * (2 * num_freqs)       # 60
        views_ch_encoded = input_ch_views * (2 * num_freqs_views)  # 24

        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_ch_encoded, width))
        
        for _ in range(self.skip_layer - 1):
            self.layers.append(nn.Linear(width, width))
            
        self.layers.append(nn.Linear(width + input_ch_encoded, width))
        
        for _ in range(self.depth - self.skip_layer - 1):
            self.layers.append(nn.Linear(width, width))
            
        self.layers.append(nn.Linear(width, width))

        self.alpha_layer = nn.Linear(width, 1)
        self.feature_layer = nn.Linear(width, width)
        self.rgb_layer = nn.Linear(width + views_ch_encoded, 3)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, view_dirs):
        x_enc = self.pos_encoder(x)
        view_enc = self.view_encoder(view_dirs)

        h = x_enc
        for i, layer in enumerate(self.layers):
            h = self.relu(layer(h))
            if i == self.skip_layer - 1:
                h = torch.cat([h, x_enc], dim=-1)  

        density = self.softplus(self.alpha_layer(h))
        feature = self.feature_layer(h)
        feature_view = torch.cat([feature, view_enc], dim=-1)
        rgb = self.sigmoid(self.rgb_layer(feature_view))

        return rgb, density

def create_nerf_model(device='cuda'):
    model = NeRFModel(depth=8, width=256, input_ch=3, input_ch_views=3,
                      num_freqs=10, num_freqs_views=4)
    return model.to(device)