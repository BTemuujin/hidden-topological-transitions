import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def stitch_comparison(ref_path, replica_path, output_path, title_ref="Reference", title_replica="Replica"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Load images
    img_ref = mpimg.imread(ref_path)
    img_rep = mpimg.imread(replica_path)

    # Display Reference
    axes[0].imshow(img_ref)
    axes[0].set_title(title_ref, fontsize=16)
    axes[0].axis('off')

    # Display Replica
    axes[1].imshow(img_rep)
    axes[1].set_title(title_replica, fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved comparison to {output_path}")

results_dir = "/mnt/c/Users/TBayaraa/Desktop/SrFeO3/GPD_follow_theory_work/results/"
figures = [4, 5, 6]

for fig_num in figures:
    ref = os.path.join(results_dir, f"PRB_Fig{fig_num}_Ref.png")
    rep = os.path.join(results_dir, f"PRB_Fig{fig_num}_Replica.png")
    out = os.path.join(results_dir, f"Comparison_Fig{fig_num}.png")

    if os.path.exists(ref) and os.path.exists(rep):
        stitch_comparison(ref, rep, out)
    else:
        print(f"Skipping Fig {fig_num}: Missing files.")
