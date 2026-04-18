import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom

class GradCAM3D:
    def __init__(self, model, target_layers):
        """
        model: Trained MultiBranchHybridSSLModel
        target_layers: Dictionary mapping modality names to layer objects
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {name: None for name in target_layers}
        self.activations = {name: None for name in target_layers}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook

        for name, layer in self.target_layers.items():
            self.hooks.append(layer.register_forward_hook(forward_hook(name)))
            self.hooks.append(layer.register_full_backward_hook(backward_hook(name)))

    def generate_heatmap(self, input_tensor, target_modality):
        """
        input_tensor: [1, 4, 64, 64, 64]
        target_modality: 'flair', 't1ce', etc.
        """
        self.model.zero_grad()
        
        # Forward pass
        embeddings, _ = self.model(input_tensor)
        
        # Target: Mean of the embedding (captures overall feature importance)
        # In a retrieval context, we want to see what makes this patient "unique"
        score = embeddings.mean()
        score.backward()

        # Get gradients and activations for the modality branch
        grads = self.gradients[target_modality] # [1, C, D, H, W]
        acts = self.activations[target_modality] # [1, C, D, H, W]

        # Global Average Pooling of gradients
        weights = torch.mean(grads, dim=(2, 3, 4), keepdim=True) # [1, C, 1, 1, 1]
        
        # Weighted sum of activations
        cam = torch.sum(weights * acts, dim=1, keepdim=True) # [1, 1, D', H', W']
        
        # ReLU to keep positive influence only
        cam = F.relu(cam)
        
        # Upscale to input resolution
        _, _, d, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(d, h, w), mode='trilinear', align_corners=False)
        cam = cam.squeeze() # [D, H, W]

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def overlay_heatmap(volume, heatmap, slice_idx=None, alpha=0.4):
    """
    volume: [D, H, W] 
    heatmap: [D, H, W] (0.0 to 1.0)
    """
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
        
    vol_slice = volume[slice_idx]
    heat_slice = heatmap[slice_idx]
    
    # Normalize volume slice for display
    vol_slice = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min() + 1e-8)
    
    plt.imshow(vol_slice, cmap='gray')
    plt.imshow(heat_slice, cmap='jet', alpha=alpha)
    plt.axis('off')

def save_explainability_panel(query_vol, query_cam, match_vol, match_cam, modality_name, save_path):
    """
    Saves a 2x2 comparison panel.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Slice selection: Max activation slice in query
    d, h, w = query_cam.shape
    slice_idx = np.unravel_index(np.argmax(query_cam), query_cam.shape)[0]
    
    # Query Original
    axes[0].imshow(query_vol[slice_idx], cmap='gray')
    axes[0].set_title(f"Query ({modality_name.upper()})")
    axes[0].axis('off')
    
    # Query + CAM
    axes[1].imshow(query_vol[slice_idx], cmap='gray')
    axes[1].imshow(query_cam[slice_idx], cmap='jet', alpha=0.5)
    axes[1].set_title("AI Attention (Heatmap)")
    axes[1].axis('off')
    
    # Match Original
    # For match, use its own max activation slice
    m_slice_idx = np.unravel_index(np.argmax(match_cam), match_cam.shape)[0]
    axes[2].imshow(match_vol[m_slice_idx], cmap='gray')
    axes[2].set_title(f"Match ({modality_name.upper()})")
    axes[2].axis('off')
    
    # Match + CAM
    axes[3].imshow(match_vol[m_slice_idx], cmap='gray')
    axes[3].imshow(match_cam[m_slice_idx], cmap='jet', alpha=0.5)
    axes[3].set_title("Match Attention")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Panel saved to: {save_path}")
