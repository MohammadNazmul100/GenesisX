import torch
from models import GenesisXModel
from Genesis.src.utitls.visualization import visualize_attention, plot_attention_statistics

def initialize_model():
    model = GenesisXModel()
    return model

if __name__ == "__main__":
    model = initialize_model()
    print(model)
    
    sample_input = torch.randint(0, 5000, (1,10))
    visualize_attention(model, sample_input)
    plot_attention_statistics(model, sample_input)