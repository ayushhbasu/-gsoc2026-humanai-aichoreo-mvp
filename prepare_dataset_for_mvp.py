"""
Dataset Preparation for GSoC 2026 HumanAI MVP
Ayush Basu | Università di Bologna

This script creates a realistic dance dataset that matches the exact
requirements of your MVP code: (50 sequences, 16 frames, 25 joints, 3D)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_realistic_dance_dataset(num_sequences=50, seq_length=16, num_joints=25):
    """
    Create realistic dance poses with human-like joint relationships.

    Joint indices (simplified COCO-style format):
    0-4: Head/neck
    5-8: Shoulders/arms
    9-12: Elbows/wrists
    13-16: Hips/legs
    17-20: Knees/ankles
    21-24: Feet/toes
    """

    joint_groups = {
        'head': list(range(0, 5)),
        'left_arm': [5, 6, 9, 10],
        'right_arm': [7, 8, 11, 12],
        'torso': [13, 14, 15, 16],
        'left_leg': [17, 18, 21, 22],
        'right_leg': [19, 20, 23, 24]
    }

    base_pose = np.zeros((num_joints, 3))

    base_pose[0] = [0, 1.6, 0]
    base_pose[1] = [0, 1.5, 0]
    base_pose[2] = [0, 1.4, 0]
    base_pose[3] = [0.1, 1.45, 0.1]
    base_pose[4] = [-0.1, 1.45, -0.1]

    base_pose[5] = [0.2, 1.35, 0]
    base_pose[6] = [0.15, 1.35, 0]
    base_pose[7] = [-0.2, 1.35, 0]
    base_pose[8] = [-0.15, 1.35, 0]

    base_pose[9] = [0.3, 1.2, 0]
    base_pose[10] = [0.4, 1.0, 0]
    base_pose[11] = [-0.3, 1.2, 0]
    base_pose[12] = [-0.4, 1.0, 0]

    base_pose[13] = [0.1, 0.9, 0]
    base_pose[14] = [-0.1, 0.9, 0]
    base_pose[15] = [0.1, 0.85, 0]
    base_pose[16] = [-0.1, 0.85, 0]

    base_pose[17] = [0.15, 0.6, 0]
    base_pose[18] = [0.15, 0.3, 0]
    base_pose[19] = [-0.15, 0.6, 0]
    base_pose[20] = [-0.15, 0.3, 0]

    base_pose[21] = [0.15, 0.05, 0.1]
    base_pose[22] = [0.15, 0.05, -0.1]
    base_pose[23] = [-0.15, 0.05, 0.1]
    base_pose[24] = [-0.15, 0.05, -0.1]

    print("Creating realistic dance dataset...")
    print(f"   - Target: {num_sequences} sequences, {seq_length} frames each")
    print(f"   - Joints: {num_joints} (human skeleton)")

    all_sequences = []

    for seq_idx in range(num_sequences):
        sequence = []

        style = seq_idx % 4

        tempo = 0.8 + (seq_idx * 0.1) % 1.2

        for frame_idx in range(seq_length):
            progress = frame_idx / seq_length

            pose = base_pose.copy()

            if style == 0:
                for j in range(num_joints):
                    pose[j, 1] += np.sin(frame_idx * tempo + j * 0.5) * 0.2
                    pose[j, 0] += np.cos(frame_idx * tempo * 0.7 + j) * 0.15
                    pose[j, 2] += np.sin(frame_idx * tempo * 0.5 + j * 0.3) * 0.1

            elif style == 1:
                if frame_idx % 4 < 2:
                    for j in joint_groups['left_arm'] + joint_groups['right_arm']:
                        pose[j, 0] += np.random.choice([-0.3, 0.3]) * 0.5
                        pose[j, 2] += np.random.choice([-0.2, 0.2]) * 0.5

                bounce = abs(np.sin(frame_idx * tempo * 2)) * 0.15
                for j in joint_groups['torso'] + joint_groups['left_leg'] + joint_groups['right_leg']:
                    pose[j, 1] -= bounce

            elif style == 2:
                lean_x = np.sin(progress * np.pi * 2) * 0.3
                lean_z = np.cos(progress * np.pi * 2) * 0.2

                for j in range(num_joints):
                    pose[j, 0] += lean_x
                    pose[j, 2] += lean_z

                if frame_idx > seq_length // 2:
                    for j in joint_groups['left_arm']:
                        pose[j, 0] += 0.4
                        pose[j, 1] += 0.2

            else:
                for j in range(num_joints):
                    pose[j] += np.random.randn(3) * 0.02

            sway = np.sin(frame_idx * 0.5) * 0.05
            for j in joint_groups['torso'] + joint_groups['head']:
                pose[j, 2] += sway

            for j in joint_groups['left_leg'][-2:] + joint_groups['right_leg'][-2:]:
                pose[j, 1] = base_pose[j, 1]

            pose += np.random.randn(*pose.shape) * 0.01

            sequence.append(pose)

        all_sequences.append(sequence)

    return np.array(all_sequences)

def visualize_sequence(poses, seq_idx=0, save_path=None):
    """
    Visualize a sequence of poses to verify realism.
    """
    sequence = poses[seq_idx]
    num_frames = sequence.shape[0]

    fig = plt.figure(figsize=(16, 16))

    for i in range(min(num_frames, 16)):
        ax = fig.add_subplot(4, 4, i+1, projection='3d')
        frame = sequence[i]

        ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2],
                  c='blue', s=20, alpha=0.8)

        connections = [
            (0,1), (1,2), (2,5), (2,7),
            (5,6), (6,9), (9,10),
            (7,8), (8,11), (11,12),
            (5,13), (7,14),
            (13,14), (13,15), (14,16),
            (15,17), (17,18), (18,21), (18,22),
            (16,19), (19,20), (20,23), (20,24)
        ]

        for conn in connections:
            if conn[0] < len(frame) and conn[1] < len(frame):
                ax.plot([frame[conn[0], 0], frame[conn[1], 0]],
                       [frame[conn[0], 1], frame[conn[1], 1]],
                       [frame[conn[0], 2], frame[conn[1], 2]],
                       'gray', alpha=0.5)

        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([0, 1.8])
        ax.set_zlim([-0.5, 0.5])
        ax.set_title(f"Frame {i+1}")
        ax.view_init(elev=20, azim=45)

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    plt.suptitle(f"Dance Sequence {seq_idx+1}", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"   - Visualization saved to {save_path}")

    plt.show()

def save_dataset_for_mvp(poses, filename='dance_dataset.npz'):
    """
    Save dataset in format compatible with your MVP code.

    Your MVP expects poses with shape (num_sequences, seq_length, 75)
    where 75 = num_joints * 3
    """
    num_sequences, seq_length, num_joints, _ = poses.shape
    poses_flat = poses.reshape(num_sequences, seq_length, num_joints * 3)

    np.savez(filename,
             poses_3d=poses,
             poses_flat=poses_flat)

    print(f"\n✅ Dataset saved to {filename}")
    print(f"   - poses_3d shape: {poses.shape} (for visualization)")
    print(f"   - poses_flat shape: {poses_flat.shape} (for your model)")

    return poses_flat

def create_text_embeddings(num_sequences=50, embedding_dim=512):
    """
    Create synthetic text embeddings that correlate with dance styles.
    This makes the contrastive learning more meaningful.
    """
    embeddings = np.zeros((num_sequences, embedding_dim))

    for i in range(num_sequences):
        style = i % 4

        if style == 0:
            embeddings[i] = np.sin(np.linspace(0, 2*np.pi, embedding_dim)) * 0.5
        elif style == 1:
            embeddings[i] = np.cos(np.linspace(0, 4*np.pi, embedding_dim)) * 0.5
        elif style == 2:
            embeddings[i] = np.random.randn(embedding_dim) * 0.3
        else:
            embeddings[i] = np.ones(embedding_dim) * 0.1

        embeddings[i] += np.random.randn(embedding_dim) * 0.1

    return embeddings

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DANCE DATASET PREPARATION FOR GSoC 2026 MVP")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)

    poses_3d = create_realistic_dance_dataset(
        num_sequences=50,
        seq_length=16,
        num_joints=25
    )

    print(f"\n✅ Generated dataset with shape: {poses_3d.shape}")
    print(f"   - {poses_3d.shape[0]} dance sequences")
    print(f"   - {poses_3d.shape[1]} frames per sequence")
    print(f"   - {poses_3d.shape[2]} joints per frame")
    print(f"   - 3D coordinates per joint")

    poses_flat = save_dataset_for_mvp(poses_3d, 'data/dance_dataset.npz')

    text_embeddings = create_text_embeddings(50, 512)
    np.save('data/text_embeddings.npy', text_embeddings)
    print(f"\n✅ Text embeddings saved to data/text_embeddings.npy")
    print(f"   - Shape: {text_embeddings.shape}")

    print("\n📊 Creating visualization of first sequence...")
    visualize_sequence(poses_3d, seq_idx=0, save_path='data/sample_sequence.png')

    with open('data/README.md', 'w') as f:
        f.write("""# Dance Dataset for GSoC 2026 HumanAI MVP

- **Sequences**: 50 dance sequences
- **Frames per sequence**: 16 frames
- **Joints per frame**: 25 joints (simplified human skeleton)
- **Coordinates**: 3D (x, y, z)

- `dance_dataset.npz` - Main dataset file
  - `poses_3d`: (50, 16, 25, 3) - For visualization
  - `poses_flat`: (50, 16, 75) - For model input
- `text_embeddings.npy` - (50, 512) synthetic text embeddings
- `sample_sequence.png` - Visualization of first sequence

```python
import numpy as np

data = np.load('data/dance_dataset.npz')
poses_flat = data['poses_flat']

poses_for_model = poses_flat.reshape(-1, 75)
```
""")
