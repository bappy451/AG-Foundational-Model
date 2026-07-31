import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import shutil
import os

# Set font family and modern styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
fig.patch.set_facecolor('white')

# Categories and styling
categories = ['Scratch\n(Random Init)', 'ImageNet-21k\n(Supervised)', 'Agri-DINO\n(Ours, SSL)']
colors = ['#E64B35', '#4DBBD5', '#00A087']
edge_colors = ['#B3311E', '#2A8E9D', '#006C5B']

# ==========================================
# PANEL A: Corn Kernel Counting MAE
# ==========================================
vals_a = [131.94, 130.97, 121.74]
x = np.arange(len(categories))
width = 0.46

bars1 = ax1.bar(x, vals_a, width=width, color=colors, edgecolor=edge_colors, linewidth=2.0, alpha=0.92, zorder=3)

# Grid and axes
ax1.set_ylim(115, 138)
ax1.set_ylabel('Validation MAE (↓, lower is better)', fontsize=13, fontweight='bold', labelpad=10)
ax1.set_title('(a) Dense Spatial Corn Kernel Counting MAE', fontsize=14, fontweight='bold', pad=14)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelsize=11.5)
ax1.grid(axis='y', linestyle='--', alpha=0.45, color='#888888', zorder=0)
ax1.set_axisbelow(True)

# Add value callouts on bars
for bar, val in zip(bars1, vals_a):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, height + 0.6,
             f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color='#111111')

# Horizontal baseline line (ImageNet-21k)
ax1.axhline(130.97, color='#2A8E9D', linestyle='--', linewidth=1.8, alpha=0.85, zorder=2)

# Horizontal reference line for Agri-DINO level in the gap
ax1.plot([1.50, 2.0], [121.74, 121.74], color='#006C5B', linestyle=':', linewidth=1.8, zorder=2)

# Improvement bracket / arrow in open space between bar 1 and bar 2
ax1.annotate('', xy=(1.50, 121.74), xytext=(1.50, 130.97),
            arrowprops=dict(arrowstyle='<->', color='#006C5B', lw=2.2))
ax1.text(1.50, 126.35, ' -7.0% Error\n (-9.23 MAE)', 
         fontsize=11.5, fontweight='bold', color='#006C5B',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#00A087', lw=1.8),
         ha='center', va='center', zorder=5)

# ==========================================
# PANEL B: PlantSeg OBB Localization Loss
# ==========================================
vals_b = [1.86, 1.78, 1.76]

bars2 = ax2.bar(x, vals_b, width=width, color=colors, edgecolor=edge_colors, linewidth=2.0, alpha=0.92, zorder=3)

ax2.set_ylim(1.68, 1.93)
ax2.set_ylabel('Val OBB Localization Loss (10^-3, ↓, lower is better)', fontsize=13, fontweight='bold', labelpad=10)
ax2.set_title('(b) PlantSeg OBB Disease Lesion Localization Loss', fontsize=14, fontweight='bold', pad=14)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelsize=11.5)
ax2.grid(axis='y', linestyle='--', alpha=0.45, color='#888888', zorder=0)
ax2.set_axisbelow(True)

for bar, val in zip(bars2, vals_b):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, height + 0.007,
             f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold', color='#111111')

# Horizontal baseline line (ImageNet-21k)
ax2.axhline(1.78, color='#2A8E9D', linestyle='--', linewidth=1.8, alpha=0.85, zorder=2)

# Horizontal reference line for Agri-DINO level in the gap
ax2.plot([1.50, 2.0], [1.76, 1.76], color='#006C5B', linestyle=':', linewidth=1.8, zorder=2)

# Improvement bracket / arrow in open space between bar 1 and bar 2
ax2.annotate('', xy=(1.50, 1.76), xytext=(1.50, 1.78),
            arrowprops=dict(arrowstyle='<->', color='#006C5B', lw=2.2))
ax2.text(1.50, 1.835, ' -5.4% Localization\n Error', 
         fontsize=11.5, fontweight='bold', color='#006C5B',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#00A087', lw=1.8),
         ha='center', va='center', zorder=5)

# Add a guiding arrow from the badge down to the gap
ax2.annotate('', xy=(1.50, 1.785), xytext=(1.50, 1.810),
            arrowprops=dict(arrowstyle='->', color='#006C5B', lw=1.8))

plt.tight_layout(pad=2.5)
pdf_path = 'fig2_functional_dissociation.pdf'
png_path = 'fig2_functional_dissociation.png'
plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
plt.savefig(png_path, bbox_inches='tight', dpi=300)

# Copy to artifacts directory
artifact_dir = r'C:\Users\mza0288\.gemini\antigravity\brain\73012caa-f1a6-4a0e-a6eb-5958837f8993'
if os.path.exists(artifact_dir):
    shutil.copy(png_path, os.path.join(artifact_dir, 'fig2_functional_dissociation.png'))
    print(f"Copied {png_path} to artifact directory!")

print("Figure 2 successfully generated as fig2_functional_dissociation.pdf and fig2_functional_dissociation.png!")
