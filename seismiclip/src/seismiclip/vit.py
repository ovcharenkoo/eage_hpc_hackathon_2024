import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Preprocessing
import torch
import matplotlib.pyplot as plt

def create_image_patches(image, patch_size):
    batch_size, channels, height, width = image.size()
    patches = []
    for h in range(0, height, patch_size):
        for w in range(0, width, patch_size):
            patch = image[:, :, h:h+patch_size, w:w+patch_size]
            patches.append(patch)
    patches = torch.cat(patches, dim=0)
    return patches

def plot_patches(patches):
    num_patches = patches.size(0)
    n = int(np.sqrt(num_patches))
    plt.figure(figsize=(2*n, 2*n))
    for i in range(num_patches):
        patch = patches[i].permute(1, 2, 0).numpy()
        plt.subplot(n, n, i + 1)
        plt.imshow(patch)
        plt.title(f"Patch {i+1}")
        plt.axis('off')
    plt.show()
    
def create_batch_patches(input_batch_image, patch_size = 8):
    batch_patches = []
    for i in range(input_batch_image.shape[0]):
        input_image = input_batch_image[i].unsqueeze(0)
        patches = create_image_patches(input_image, patch_size)
        batch_patches.append(patches)
    return torch.stack(batch_patches, dim=0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention, V)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.fc_out(attended_values)
        return output, attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
                                    nn.Linear(d_model, dim_feedforward),
                                    nn.ReLU(),
                                    nn.Linear(dim_feedforward, d_model),
                                )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended, attention = self.attention(x, x, x, mask)
        x = x + self.dropout1(self.norm1(attended))
        feedforward = self.fc(x)
        x = x + self.dropout2(self.norm2(feedforward))
        return x, attention

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, d_model, n_heads, image_embedding_size, dim_feedforward, num_layers, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        
        self.patch_dim = 3 * patch_size ** 2
        self.patch_embeddings = nn.Linear(self.patch_dim, d_model)
        self.position_embeddings = self.generate_positional_encodings(num_patches + 1, d_model) # nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.fc_head = nn.Linear(d_model, image_embedding_size)
        # self.fc = nn.Linear(image_embedding_size, 10)
        
    def generate_positional_encodings(self, num_patches, d_model):
        position_encodings = torch.zeros(1, num_patches, d_model)  # Add +1 to num_patches
        position = torch.arange(0, num_patches, dtype=torch.float32).unsqueeze(1)  # Add +1 to num_patches
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        position_encodings[:, :, 0::2] = torch.sin(position * div_term)
        position_encodings[:, :, 1::2] = torch.cos(position * div_term)
        return position_encodings

        
    def create_image_patches(self, image):
        batch_size, channels, height, width = image.size()
        patches = []
        for h in range(0, height, self.patch_size):
            for w in range(0, width, self.patch_size):
                patch = image[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patches.append(patch)
        patches = torch.cat(patches, dim=0)
        return patches
    
    def create_batch_patches(self,input_batch_image):
        batch_patches = []
        for i in range(input_batch_image.shape[0]):
            input_image = input_batch_image[i].unsqueeze(0)
            patches = self.create_image_patches(input_image)
            batch_patches.append(patches)
        return torch.stack(batch_patches, dim=0).float()
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.create_batch_patches(x)
        x = x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3)*x.size(4))
        x = self.patch_embeddings(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings.to(x.device)
        for transformer in self.transformer_blocks:
            x, _ = transformer(x)

        x = x[:, 0]
        x = self.dropout(x)
        x = self.fc_head(x)
        # x = self.fc(x)
        return x