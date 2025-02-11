import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Define a custom colormap: Black -> Green -> Yellow -> Red
colors = ["black", "green", "yellow", "red"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

def get_attention_patterns(model, input_tokens):
    """Extract attention patterns from all layers."""
    with torch.no_grad():
        token_embeddings = model.embedding_layer(input_tokens)
        return [layer.self_attn(token_embeddings, token_embeddings, token_embeddings, need_weights=True)[1].squeeze(0).cpu().numpy()
                for layer in model.encoder_stack.layers]

def visualize_attention(model, input_tokens, sequence_length=10):
    """Visualize attention patterns for each layer."""
    layer_attention_patterns = get_attention_patterns(model, input_tokens)
    fig, axes = plt.subplots(1, len(layer_attention_patterns), figsize=(15, 4))
    if len(layer_attention_patterns) == 1: axes = [axes]
    
    word_positions = [f'Word {i+1}' for i in range(sequence_length)]
    for layer_idx, axis in enumerate(axes):
        sns.heatmap(layer_attention_patterns[layer_idx], ax=axis, cmap=cmap, square=True,
                    xticklabels=word_positions, yticklabels=word_positions, annot=False, cbar=False)
        axis.set_title(f'Layer {layer_idx + 1}', fontsize=12)
        axis.set_xlabel('To Word', fontsize=10)
        axis.set_ylabel('From Word', fontsize=10)
        axis.tick_params(labelsize=8)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(axes[0].collections[0], cax=cbar_ax, label='Attention Strength')
    plt.suptitle('How Words Pay Attention to Each Other', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

def plot_attention_statistics(model, input_tokens):
    """Show summary charts about word connections."""
    layer_attention_patterns = get_attention_patterns(model, input_tokens)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    avg_attention = [np.mean(layer) for layer in layer_attention_patterns]
    axes[0].bar(range(1, len(avg_attention) + 1), avg_attention, color='skyblue')
    axes[0].set_title('How Much Words Care on Average')
    axes[0].set_xlabel('Step Number')
    axes[0].set_ylabel('Average Caring')
    
    all_attention = np.concatenate([layer.flatten() for layer in layer_attention_patterns])
    sns.histplot(all_attention, ax=axes[1], bins=30, color='lightgreen')
    axes[1].set_title('How Common Each Level of Caring Is')
    axes[1].set_xlabel('Caring Level')
    axes[1].set_ylabel('How Many Times It Happens')
    
    plt.tight_layout()
    plt.show()