
import torch
import sys
import os

sys.path.append(os.getcwd())

from models.groupmamba import GroupMamba

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Instantiate GroupMamba
    # Using tiny config or similar
    try:
        model = GroupMamba(
            stem_hidden_dim = 32,
            embed_dims = [64, 128, 348, 448], 
            mlp_ratios = [8, 8, 4, 4],
            depths = [3, 4, 9, 3] # Tiny
        ).to(device)
        print("GroupMamba Instantiated.")
    except Exception as e:
        print(f"Instantiation Failed: {e}")
        return

    # Dummy Input (B, 3, 224, 224) - GroupMamba is a Vision Model by default?
    # Let's check input expectations. It usually expects images.
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    try:
        output = model(input_tensor)
        if isinstance(output, tuple):
             print(f"Forward Pass Successful. Output is Tuple.")
             for i, item in enumerate(output):
                 if hasattr(item, 'shape'):
                     print(f"  Item {i} Shape: {item.shape}")
                 else:
                     print(f"  Item {i}: {type(item)}")
        else:
             print(f"Forward Pass Successful. Output Shape: {output.shape}")
    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        # traceback
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
