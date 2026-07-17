"""
NeRF with Hash Grid Encoding (Instant-NGP style)
This implementation demonstrates heterogeneous workload:
  - sample_position: Simple ray sampling (lightweight)
  - hash_encode: Memory-intensive hash table lookups (DOE target)
  - nerf_mlp: Compute-intensive MLP (CGRA target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Part 1: Sample Position (Ray Sampling)
# ============================================================================
class RaySampler(nn.Module):
    """Sample 3D positions along rays"""
    def __init__(self, num_samples=64, near=2.0, far=6.0):
        super().__init__()
        self.num_samples = num_samples
        self.near = near
        self.far = far
    
    def forward(self, rays_o, rays_d):
        """
        Args:
            rays_o: ray origins [batch_size, 3]
            rays_d: ray directions [batch_size, 3]
        Returns:
            positions: sampled 3D positions [batch_size, num_samples, 3]
        """
        batch_size = rays_o.shape[0]
        
        # Uniform sampling along rays
        t_vals = torch.linspace(self.near, self.far, self.num_samples, 
                               device=rays_o.device, dtype=rays_o.dtype)
        t_vals = t_vals.unsqueeze(0).expand(batch_size, -1)  # [B, N]
        
        # positions = rays_o + t * rays_d
        positions = rays_o.unsqueeze(1) + t_vals.unsqueeze(2) * rays_d.unsqueeze(1)
        
        return positions  # [batch_size, num_samples, 3]


# ============================================================================
# Part 2: Hash Grid Encoding (Memory-Intensive, DOE Target)
# ============================================================================
class HashGridEncoder(nn.Module):
    """
    Multi-resolution hash encoding (Instant-NGP style)
    This is memory-intensive: lots of hash table lookups
    """
    def __init__(self, 
                 num_levels=16,
                 features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 finest_resolution=512):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        
        # Compute resolution per level
        self.b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (num_levels - 1))
        
        # Hash tables for each level (learnable parameters)
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(2**log2_hashmap_size, features_per_level) * 0.01)
            for _ in range(num_levels)
        ])
    
    def hash_function(self, coords, level):
        """
        Hash 3D integer coordinates to hash table index
        Simple hash: (x * 1) ^ (y * 2654435761) ^ (z * 805459861) % table_size
        """
        primes = torch.tensor([1, 2654435761, 805459861], 
                             device=coords.device, dtype=torch.long)
        hashed = torch.sum(coords * primes, dim=-1)
        return torch.bitwise_and(hashed, (2**self.log2_hashmap_size - 1))
    
    def grid_sample_3d(self, positions, level):
        """
        Sample features from hash grid at given level
        positions: [batch_size, num_samples, 3], in range [0, 1]
        """
        batch_size, num_samples, _ = positions.shape
        resolution = int(np.floor(self.base_resolution * (self.b ** level)))
        
        # Scale positions to grid resolution
        scaled_pos = positions * (resolution - 1)  # [B, N, 3]
        
        # Get integer grid coordinates (8 corners of cube)
        base_coords = torch.floor(scaled_pos).long()  # [B, N, 3]
        
        # Trilinear interpolation weights
        frac = scaled_pos - base_coords.float()  # [B, N, 3]
        
        # Flatten batch and samples for processing
        base_coords_flat = base_coords.view(-1, 3)  # [B*N, 3]
        frac_flat = frac.view(-1, 3)  # [B*N, 3]
        
        # Sample from 8 corners
        features_list = []
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    offset = torch.tensor([dx, dy, dz], device=positions.device, dtype=torch.long)
                    corner_coords = base_coords_flat + offset  # [B*N, 3]
                    
                    # Hash coordinates to table indices
                    indices = self.hash_function(corner_coords, level)  # [B*N]
                    
                    # Lookup features from hash table
                    corner_features = self.hash_tables[level][indices]  # [B*N, F]
                    
                    # Compute trilinear weight
                    weight = 1.0
                    weight *= (1 - frac_flat[:, 0]) if dx == 0 else frac_flat[:, 0]
                    weight *= (1 - frac_flat[:, 1]) if dy == 0 else frac_flat[:, 1]
                    weight *= (1 - frac_flat[:, 2]) if dz == 0 else frac_flat[:, 2]
                    
                    features_list.append(corner_features * weight.unsqueeze(1))
        
        # Sum contributions from all corners
        interpolated_features = sum(features_list)  # [B*N, F]
        
        # Reshape back
        interpolated_features = interpolated_features.view(
            batch_size, num_samples, self.features_per_level
        )
        
        return interpolated_features
    
    def forward(self, positions):
        """
        Encode 3D positions with multi-resolution hash encoding
        Args:
            positions: [batch_size, num_samples, 3], in range [0, 1]
        Returns:
            encoded: [batch_size, num_samples, num_levels * features_per_level]
        """
        # Normalize positions to [0, 1]
        positions_normalized = (positions + 1.0) / 2.0  # Assume input in [-1, 1]
        
        # Sample from each resolution level
        encoded_list = []
        for level in range(self.num_levels):
            level_features = self.grid_sample_3d(positions_normalized, level)
            encoded_list.append(level_features)
        
        # Concatenate all levels
        encoded = torch.cat(encoded_list, dim=-1)  # [B, N, L*F]
        
        return encoded


# ============================================================================
# Part 3: NeRF MLP (Compute-Intensive, CGRA Target)
# ============================================================================
class NeRFMLP(nn.Module):
    """
    MLP to predict density and color from encoded features
    This is compute-intensive: matrix multiplications and activations
    """
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Density network (with feature extraction)
        density_layers = []
        in_dim = input_dim
        for i in range(num_layers):
            density_layers.append(nn.Linear(in_dim, hidden_dim))
            density_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.density_features_net = nn.Sequential(*density_layers)
        self.density_head = nn.Linear(hidden_dim, 1)  # Output: density
        
        # Color network (takes density features + view direction encoding)
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # RGB in [0, 1]
        )
    
    def forward(self, encoded_features, view_dirs):
        """
        Args:
            encoded_features: [batch_size, num_samples, input_dim]
            view_dirs: [batch_size, 3]
        Returns:
            density: [batch_size, num_samples, 1]
            rgb: [batch_size, num_samples, 3]
        """
        batch_size, num_samples, _ = encoded_features.shape
        
        # Pass through density network to get features
        density_features = self.density_features_net(encoded_features)  # [B, N, hidden_dim]
        
        # Predict density from features
        density_output = self.density_head(density_features)  # [B, N, 1]
        
        # Expand view directions for all samples
        view_dirs_expanded = view_dirs.unsqueeze(1).expand(-1, num_samples, -1)
        
        # Concatenate density features and view direction
        color_input = torch.cat([density_features, view_dirs_expanded], dim=-1)
        
        # Predict RGB
        rgb = self.color_net(color_input)  # [B, N, 3]
        
        # Apply ReLU to density
        density = F.relu(density_output)  # [B, N, 1]
        
        return density, rgb


# ============================================================================
# Complete NeRF Model
# ============================================================================
class HashGridNeRF(nn.Module):
    """Complete NeRF model with hash grid encoding"""
    def __init__(self,
                 num_samples=64,
                 num_levels=16,
                 features_per_level=2,
                 hidden_dim=64):
        super().__init__()
        
        # Three main components
        self.sampler = RaySampler(num_samples=num_samples)
        self.encoder = HashGridEncoder(
            num_levels=num_levels,
            features_per_level=features_per_level
        )
        self.mlp = NeRFMLP(
            input_dim=num_levels * features_per_level,
            hidden_dim=hidden_dim
        )
    
    def forward(self, rays_o, rays_d):
        """
        Forward pass: sample -> encode -> MLP
        Args:
            rays_o: [batch_size, 3]
            rays_d: [batch_size, 3]
        Returns:
            density: [batch_size, num_samples, 1]
            rgb: [batch_size, num_samples, 3]
        """
        # Part 1: Sample positions (lightweight)
        positions = self.sampler(rays_o, rays_d)  # [B, N, 3]
        
        # Part 2: Hash encoding (memory-intensive, DOE target)
        encoded = self.encoder(positions)  # [B, N, L*F]
        
        # Part 3: MLP (compute-intensive, CGRA target)
        density, rgb = self.mlp(encoded, rays_d)  # [B, N, 1], [B, N, 3]
        
        return density, rgb


# ============================================================================
# Simplified Volume Rendering
# ============================================================================
def volume_rendering(density, rgb, num_samples, near=2.0, far=6.0):
    """
    Simple volume rendering (for completeness)
    Args:
        density: [batch_size, num_samples, 1]
        rgb: [batch_size, num_samples, 3]
    Returns:
        rendered_rgb: [batch_size, 3]
    """
    # Compute delta (distance between samples)
    delta = (far - near) / num_samples
    
    # Compute alpha (opacity)
    alpha = 1.0 - torch.exp(-density * delta)  # [B, N, 1]
    
    # Compute transmittance
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    transmittance = torch.cat([
        torch.ones_like(transmittance[:, :1]),
        transmittance[:, :-1]
    ], dim=1)  # [B, N, 1]
    
    # Weighted sum
    weights = alpha * transmittance  # [B, N, 1]
    rendered_rgb = torch.sum(weights * rgb, dim=1)  # [B, 3]
    
    return rendered_rgb


# ============================================================================
# Test/Demo Code
# ============================================================================
def test_nerf_hash_grid():
    """Test the NeRF hash grid model"""
    device = torch.device('cpu')  # Use CPU for MLIR export
    
    # Create model
    model = HashGridNeRF(
        num_samples=64,
        num_levels=8,  # Reduced for testing
        features_per_level=2,
        hidden_dim=32
    ).to(device)
    
    # Create dummy input
    batch_size = 4
    rays_o = torch.randn(batch_size, 3, device=device)
    rays_d = torch.randn(batch_size, 3, device=device)
    rays_d = F.normalize(rays_d, dim=-1)  # Normalize direction
    
    # Forward pass
    print("Running forward pass...")
    density, rgb = model(rays_o, rays_d)
    
    print(f"Input shape: rays_o {rays_o.shape}, rays_d {rays_d.shape}")
    print(f"Output shape: density {density.shape}, rgb {rgb.shape}")
    
    # Volume rendering
    rendered = volume_rendering(density, rgb, num_samples=64)
    print(f"Rendered RGB shape: {rendered.shape}")
    
    return model, rays_o, rays_d


if __name__ == "__main__":
    # Run test
    model, rays_o, rays_d = test_nerf_hash_grid()
    print("\n✓ NeRF Hash Grid model test passed!")
    
    # Export to TorchScript (preparation for MLIR)
    print("\nExporting to TorchScript...")
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.save("nerf_hash_grid_scripted.pt")
        print("✓ TorchScript export successful!")
    except Exception as e:
        print(f"TorchScript export failed (expected): {e}")
        print("Note: Hash operations may need special handling for MLIR export")
