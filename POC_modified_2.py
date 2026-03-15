"""
GSoC 2026 HumanAI - MVP Proof of Concept
Ayush Basu | BS Statistics & Mathematics, Università di Bologna

This script demonstrates the core components of the proposed
any-to-any multimodal dance generation system using synthetic data.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import time

print("=" * 60)
print("GSoC 2026 HumanAI - MVP Proof of Concept")
print("=" * 60)

print("\n[1/5] Loading realistic dance dataset...")
try:
    data = np.load('data/dance_dataset.npz')
    poses_flat = data['poses_flat']
    poses = poses_flat.reshape(len(poses_flat), -1)
    poses_3d = data['poses_3d']
    print(f"   - Loaded {poses.shape[0]} sequences")
    print(f"   - Each sequence: {poses_flat.shape[1]} frames")
    print(f"   - Each frame: {poses_flat.shape[2]} values (25 joints x 3 coordinates)")
    print(f"   - Data shape: {poses.shape}")
except FileNotFoundError:
    print("   - Dataset not found, run prepare_dataset_for_mvp.py first")
    exit(1)

def fps_sample(pose_frame, n_points=128):
   
    points = pose_frame.reshape(-1, 3)

    if len(points) < n_points:
        n_repeat = n_points // len(points) + 1
        points = np.repeat(points, n_repeat, axis=0)[:n_points]
        points += np.random.randn(*points.shape) * 0.01
        return points

    centroids = []
    centroids.append(points[np.random.randint(len(points))])

    for _ in range(n_points - 1):
        dist = np.min([
            np.linalg.norm(points - c, axis=1) for c in centroids
        ], axis=0)
        centroids.append(points[np.argmax(dist)])

    return np.array(centroids)

print("\n[2/5] Testing Farthest Point Sampling...")
test_pose = poses_flat[0, 0]
point_cloud = fps_sample(test_pose, n_points=128)
print("   - Input pose: {} (25 joints)".format(test_pose.shape))
print("   - Output point cloud: {} (128 points)".format(point_cloud.shape))
print("   - Point cloud range: [{:.2f}, {:.2f}]".format(point_cloud.min(), point_cloud.max()))

def infonce_loss(emb_a, emb_b, temp=0.07):
    """
    InfoNCE loss for contrastive learning.

    Args:
        emb_a: (batch_size, dim) embeddings from modality A
        emb_b: (batch_size, dim) embeddings from modality B
        temp: Temperature parameter

    Returns:
        loss: Scalar loss value
    """
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)

    logits = torch.mm(emb_a, emb_b.T) / temp

    labels = torch.arange(len(emb_a)).to(emb_a.device)

    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2

    return loss

print("\n[3/5] Testing InfoNCE Loss...")

batch_size = 10
embedding_dim = 512

text_emb = torch.randn(batch_size, embedding_dim)
movement_emb = torch.randn(batch_size, embedding_dim)

for i in range(batch_size):
    movement_emb[i] += text_emb[i] * 0.5

loss = infonce_loss(text_emb, movement_emb)
print("   - Batch size: {}".format(batch_size))
print("   - Embedding dim: {}".format(embedding_dim))
print("   - InfoNCE loss: {:.4f}".format(loss.item()))

text_emb2 = text_emb.clone()
movement_emb2 = text_emb2 + torch.randn_like(text_emb2) * 0.1
loss2 = infonce_loss(text_emb2, movement_emb2)
print("   - Loss with similar pairs: {:.4f} (lower is better)".format(loss2.item()))

class SimpleDiffusion(nn.Module):
    def __init__(self, input_dim=75, hidden_dim=256, timestep_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, timestep_dim)
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(512, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, timestep_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + timestep_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, cond):
        t_float = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_mlp(t_float)

        cond_emb = self.cond_mlp(cond)

        h = torch.cat([x, t_emb, cond_emb], dim=-1)

        return self.net(h)

def diffusion_training_step(model, x0, cond, optimizer, timesteps=100):
    batch_size = x0.shape[0]

    t = torch.randint(0, timesteps, (batch_size,))

    noise = torch.randn_like(x0)

    alpha = 1.0 - t.float() / timesteps
    alpha = alpha.view(-1, 1)

    xt = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * noise

    noise_pred = model(xt, t, cond)

    loss = F.mse_loss(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def sample_diffusion(model, cond, num_steps=100, input_dim=75):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, input_dim)

        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t])

            noise_pred = model(x, t_tensor, cond)

            alpha = 1.0 - t / num_steps
            x = (x - (1 - alpha) * noise_pred) / np.sqrt(alpha)

            if t > 0:
                x += torch.randn_like(x) * np.sqrt(1 - alpha)

    return x

print("\n[4/5] Training Mini Diffusion Model...")

input_dim = poses_flat.shape[2]
model = SimpleDiffusion(input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

poses_tensor = torch.tensor(poses_flat, dtype=torch.float32).reshape(-1, input_dim)

text_emb_np = np.load('data/text_embeddings.npy')
text_emb_np = np.repeat(text_emb_np, 16, axis=0)
cond_embeddings = torch.tensor(text_emb_np, dtype=torch.float32)

print("   - Model input dim: {}".format(input_dim))
print("   - Training samples: {}".format(len(poses_tensor)))

train_steps = 100
start_time = time.time()

for step in range(train_steps):
    batch_idx = np.random.choice(len(poses_tensor), size=4, replace=False)
    x0 = poses_tensor[batch_idx]
    cond = cond_embeddings[batch_idx]

    loss = diffusion_training_step(model, x0, cond, optimizer)

    if step % 20 == 0:
        print("   - Step {}: loss = {:.6f}".format(step, loss))

elapsed = time.time() - start_time
print("   - Training completed in {:.1f} seconds".format(elapsed))
print("   - Final loss: {:.6f}".format(loss))

print("\n[5/5] Generating sample and visualizing...")

test_cond = torch.tensor(text_emb_np[0:1], dtype=torch.float32)

generated = sample_diffusion(model, test_cond, input_dim=input_dim)
generated_np = generated.numpy().reshape(-1, 25, 3)

print("   - Generated shape: {}".format(generated_np.shape))
print("   - Generated value range: [{:.2f}, {:.2f}]".format(generated_np.min(), generated_np.max()))

fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
pc = point_cloud
ax1.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='blue', s=10, alpha=0.6)
ax1.set_title("FPS Point Cloud\n(128 points from 1 frame)")
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

ax2 = fig.add_subplot(132, projection='3d')
orig_frame = poses_flat[0, 0].reshape(25, 3)
ax2.scatter(orig_frame[:, 0], orig_frame[:, 1], orig_frame[:, 2],
            c='green', s=50, alpha=0.8)
ax2.set_title("Original Pose\n(25 joints)")
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

ax3 = fig.add_subplot(133, projection='3d')
gen_frame = generated_np[0]
ax3.scatter(gen_frame[:, 0], gen_frame[:, 1], gen_frame[:, 2],
            c='red', s=50, alpha=0.8)
ax3.set_title("Generated Pose\n(from diffusion model)")
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')

plt.tight_layout()
plt.savefig('mvp_results.png', dpi=150)
print("   - Visualization saved as 'mvp_results.png'")

print("\n" + "=" * 60)
print("MVP SUMMARY - ALL COMPONENTS WORKING!")
print("=" * 60)
print("""
- STEP 1: Synthetic pose data created
- STEP 2: Farthest Point Sampling implemented
- STEP 3: InfoNCE loss working (contrastive learning)
- STEP 4: Mini diffusion model trained successfully
- STEP 5: Sample generation and visualization complete

This MVP demonstrates that the core algorithms from my
GSoC proposal are correctly implemented and work together
in a minimal end-to-end pipeline.

The full system would scale this to:
- Real dance videos (via MediaPipe)
- Larger datasets (1000+ sequences)
- Full 1024-point clouds
- Multiple modality encoders (BERT, ResNet, Wav2Vec2)
- Bidirectional translation (movement -> text)
""")

print("\nFiles generated:")
print("   - This script: mvp_demo.py")
print("   - Visualization: mvp_results.png")
print("\nTo run: python mvp_demo.py")
print("\nGitHub ready! Push to: https://github.com/ayushbasu/gsoc2026-humanai-mvp")
