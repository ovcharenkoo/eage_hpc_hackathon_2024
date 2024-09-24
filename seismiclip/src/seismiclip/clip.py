import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ImageProjection(nn.Module):
    def __init__(self, image_embedding_size, shared_embedding_size, dropout = 0.1):
        super(ImageProjection, self).__init__()
        self.image_projection = nn.Linear(image_embedding_size, shared_embedding_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(shared_embedding_size)
        
    def forward(self, image_embeddings):
        projected_embeddings = self.image_projection(image_embeddings)
        
        x = self.gelu(projected_embeddings)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected_embeddings
        x = self.layer_norm(x)
        
        return x # projected_embeddings

class TextProjection(nn.Module):
    def __init__(self, text_embedding_size, shared_embedding_size, dropout = 0.1):
        super(TextProjection, self).__init__()
        self.text_projection = nn.Linear(text_embedding_size, shared_embedding_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(shared_embedding_size)
        
    def forward(self, text_embeddings):
        projected_embeddings = self.text_projection(text_embeddings)
        
        x = self.gelu(projected_embeddings)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected_embeddings
        x = self.layer_norm(x)
        
        return x # projected_embeddings
    
class CosineSimilarity(nn.Module):
    def __init__(self, temperature = 0.2 ):
        super(CosineSimilarity, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        normalized_images = nn.functional.normalize(image_embeddings, dim=-1)
        normalized_texts = nn.functional.normalize(text_embeddings, dim=-1)
        
        similarities = torch.matmul(normalized_images, normalized_texts.t())*self.temperature
        return similarities
    
class SymmetricalLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SymmetricalLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        
    def forward(self, image_embeddings, text_embeddings):
        sim_image_to_text = self.cosine_similarity(image_embeddings, text_embeddings)
        sim_text_to_image = self.cosine_similarity(text_embeddings, image_embeddings)
        
        # # Similarity Loss 
        # loss_image_to_text = torch.mean((1 sim_image_to_text).clamp(min=0))
        # loss_text_to_image = torch.mean((1 sim_text_to_image).clamp(min=0))
        
        # Cross Entropy Loss
        target = torch.ones_like(sim_image_to_text)  # Label 1 indicates alignment
        loss_image_to_text = nn.functional.binary_cross_entropy_with_logits(sim_image_to_text, target, reduction='mean')
        loss_text_to_image = nn.functional.binary_cross_entropy_with_logits(sim_text_to_image, target, reduction='mean')
        
        total_loss = loss_image_to_text + loss_text_to_image
        return total_loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # Normalize the input features
        image_features = F.normalize(image_features, dim=-1, p=2)
        text_features = F.normalize(text_features, dim=-1, p=2)

        # Calculate similarity scores (dot product)
        similarity_scores = torch.matmul(image_features, text_features.T) / self.temperature

        # Generate labels for positive and negative pairs
        batch_size = similarity_scores.size(0)
        labels = torch.arange(batch_size, device=similarity_scores.device)
        
        # Calculate contrastive loss
        loss = F.cross_entropy(similarity_scores, labels)
        return loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_clip_loss_function( text_projection,  image_projection, mode="eval", temperature_value = 1):
    logits = (text_projection @ image_projection.T) / temperature_value
    if mode=="train":
        images_similarity = image_projection @ image_projection.T
        texts_similarity = text_projection @ text_projection.T
        targets = F.softmax( (images_similarity + texts_similarity) / 2 * temperature_value, dim=-1 )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    elif mode=="eval":
        return logits
    else:
        print("Mention mode")
        return None
    
def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params

def create_image_embeddings(images, vit_model, image_projector):
    with torch.no_grad():
        image_embeddings = vit_model(images)
        image_projection = image_projector(image_embeddings)
    return image_projection

def image_retrieval_function(inputs, n ,bert_data, bert_model, text_projector, image_embeddings_list_train, clip_trainset, display=False):
    with torch.no_grad():
        
        outputs = bert_model(inputs_embeds=inputs.view(-1,bert_data.shape[-2],bert_data.shape[-1]))
        text_embeddings = outputs.hidden_states[-1].mean(dim=1)
        text_projection = text_projector(text_embeddings)
    
    similarity_scores_list = []
    for index in tqdm(range(len(image_embeddings_list_train))):
        score = torch.dot( text_projection[0], image_embeddings_list_train[index] )
        similarity_scores_list.append( score.cpu().numpy() )
    
    max_indexes = np.array(similarity_scores_list).argsort()[-n:][::-1]
    if display:
        for index in max_indexes:
            image_tensor = clip_trainset[index][1]
            plt.imshow(image_tensor[0].detach().cpu().numpy())
            plt.savefig('similar_img_'+str(index)+'.pdf')
            plt.show()
        return None
    else:
        return max_indexes