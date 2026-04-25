from PIL import Image, ImageDraw, ImageFont
import os

def stitch_comparison(ref_path, replica_path, output_path):
    # Open images
    img_ref = Image.open(ref_path).convert("RGB")
    img_rep = Image.open(replica_path).convert("RGB")

    # Ensure they have the same height for a clean stitch
    if img_ref.height != img_rep.height:
        aspect_ratio = img_rep.width / img_rep.height
        new_width = int(img_ref.height * aspect_ratio)
        img_rep = img_rep.resize((new_width, img_ref.height), Image.Resampling.LANCZOS)

    # Parameters for titles
    title_height = 80
    total_width = img_ref.width + img_rep.width
    total_height = img_ref.height + title_height

    # Create a new image with a white background
    combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

    # Paste images side-by-side, shifted down by title_height
    combined_image.paste(img_ref, (0, title_height))
    combined_image.paste(img_rep, (img_ref.width, title_height))

    # Add text
    draw = ImageDraw.Draw(combined_image)

    # Attempt to load a readable font, fallback to default
    try:
        # Common Linux font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 40)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Calculate text positions
    # Original/Reference
    text_ref = "Original"
    # Use textbbox to get size (modern PIL)
    bbox_ref = draw.textbbox((0, 0), text_ref, font=font)
    w_ref, h_ref = bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]
    draw.text((img_ref.width // 2 - w_ref // 2, (title_height - h_ref) // 2), text_ref, fill=(0, 0, 0), font=font)

    # Replica
    text_rep = "Replica"
    bbox_rep = draw.textbbox((0, 0), text_rep, font=font)
    w_rep, h_rep = bbox_rep[2] - bbox_rep[0], bbox_rep[3] - bbox_rep[1]
    draw.text((img_ref.width + img_rep.width // 2 - w_rep // 2, (title_height - h_rep) // 2), text_rep, fill=(0, 0, 0), font=font)

    # Save the result
    combined_image.save(output_path, dpi=(300, 300))
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
