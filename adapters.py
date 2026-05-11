import torch
from torch import nn
import torch.nn.functional as F
import os

class ABMIL(nn.Module):
    def __init__(self,
                     input_dim=384,
                     inner_dim=256, 
                     output_dim=1024, 
                     use_layernorm=True, 
                     dropout=0.0,
                ):
        super(ABMIL,self).__init__()

        self.inner_proj = nn.Linear(input_dim,inner_dim)
        self.output_dim = output_dim
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout)
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(inner_dim)
        self.attention_V = nn.Linear(inner_dim, inner_dim)
        self.attention_U = nn.Linear(inner_dim, inner_dim)
        self.sigmoid = nn.Sigmoid()
        self.attention_weights = nn.Linear(inner_dim, 1)

        self.fc1 = nn.Linear(inner_dim, inner_dim*2)
        self.fc2 = nn.Linear(inner_dim*2, output_dim)
   
        
    def forward(self, data):
        x = self.inner_proj(data)
        
        if self.use_layernorm:
            x = self.layernorm(x)        
        
        # Apply attention mechanism
        V = torch.tanh(self.attention_V(x))  # Shape: (batch_size, num_patches, inner_dim)
        U = self.sigmoid(self.attention_U(x))  # Shape: (batch_size, num_patches, inner_dim)
        
        # Compute attention scores
        attn_scores = self.attention_weights(V * U)  # Shape: (batch_size, num_patches, 1)
        attn_scores = torch.softmax(attn_scores, dim=1)  # Shape: (batch_size, num_patches, 1)
        
        # Weighted sum of patch features
        out = self.dropout(attn_scores * x)

        out = torch.relu(self.fc1(out))
        output = self.fc2(out)
        
        return output



class MlpBlock(nn.Module):
    def __init__(self, mlp_dim):
        super(MlpBlock, self).__init__()
        self.dense1 = nn.Linear(in_features=mlp_dim, out_features=mlp_dim)
        self.dense2 = nn.Linear(in_features=mlp_dim, out_features=mlp_dim)

    def forward(self, x):
        y = self.dense1(x)
        y = F.gelu(y)
        return self.dense2(y)

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.tokens_mlp = MlpBlock(tokens_mlp_dim)
        self.channels_mlp = MlpBlock(channels_mlp_dim)
        self.layer_norm1 = nn.LayerNorm(tokens_mlp_dim)
        self.layer_norm2 = nn.LayerNorm(tokens_mlp_dim)

    def forward(self, x):
        y = self.layer_norm1(x)
        y = y.permute(0, 2, 1)
        y = self.channels_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.layer_norm2(x)
        return x + self.tokens_mlp(y)

class MlpMixer(nn.Module):
    def __init__(self, out_dim=1024, num_blocks=1, tokens_mlp_dim=384, channels_mlp_dim=1029):
        super(MlpMixer, self).__init__()
        self.blocks = nn.Sequential(*[MixerBlock(tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(tokens_mlp_dim)
        self.fc1 = nn.Linear(tokens_mlp_dim, out_dim//2)
        self.fc2 = nn.Linear(out_dim//2, out_dim)

    def forward(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class EdgeDetector(nn.Module):
    def __init__(self, path, checkpoint = 0, device = 'cuda'):
        super(EdgeDetector, self).__init__()
        self.path = path
        self.sobel_x = (torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device)/8).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        self.sobel_y = (torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device)/8).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU()
            )
        
        if checkpoint > 0:
            self.load_cnn(checkpoint)

        self.cnn.to(device)


    def forward(self, images):
        batch_size, _, _, _ = images.shape
        edges_x = F.conv2d(images, self.sobel_x, padding=1)
        edges_y = F.conv2d(images, self.sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        out = self.cnn(edges)
        return out

    def save_cnn(self, checkpoint):
        cnn_weights = {name: param.data for name, param in self.cnn.named_parameters()}
        torch.save(cnn_weights, os.path.join(self.path, f'cnn_{checkpoint}.pth'))
    
    def load_cnn(self, checkpoint):
        cnn_weights = torch.load(os.path.join(self.path, f'cnn_{checkpoint}.pth'))
        for name, param in cnn_weights.items():
            if name in self.cnn.state_dict():
                self.cnn.state_dict()[name].copy_(param)
    
    def freeze_cnn(self):
        self.cnn.requires_grad_(False)


if __name__ == "__main__":
    tensor = torch.rand(4, 1029, 384)

    #model = ABMIL()
    model = MlpMixer()
    model.eval()
    model.requires_grad_ = False
    a = model(tensor)
    print(a.shape)