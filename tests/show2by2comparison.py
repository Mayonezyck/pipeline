import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the images
raw = mpimg.imread('data/Hamlyn/rectified01/image01/0000000025.jpg')
gt = mpimg.imread('data/Hamlyn/rectified01/depth01/0000000025.png')
da2 = mpimg.imread('output/depth-anything:v2_20241219162426_output/0000000025.png')
dac = mpimg.imread('output/endo-dac_20241219151643_output_20241219151700_postprocessed/0000000025_depth.png')
print(raw.shape, gt.shape, da2.shape, dac.shape)
# Create a figure and set of subplots
fig, axes = plt.subplots(2, 2, figsize=(10,10))

# Display the raw image (original)
axes[0, 0].imshow(raw)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Display the ground truth depth with a colormap
axes[0, 1].imshow(gt, cmap='rainbow')
axes[0, 1].set_title('Ground Truth Depth')
axes[0, 1].axis('off')

# Display the DA2 depth estimation with the same colormap
axes[1, 0].imshow(255-da2[:,:,0], cmap='rainbow')
axes[1, 0].set_title('DA2 Depth')
axes[1, 0].axis('off')

# Display the DAC depth estimation with the same colormap
axes[1, 1].imshow(dac, cmap='rainbow')
axes[1, 1].set_title('DAC Depth')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
