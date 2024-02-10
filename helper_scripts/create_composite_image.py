from PIL import Image
import os

def overlay_images(image_dir, output_path = "/scratches/dialfs/alta/relevance/ns832/data/visual_and_text/BLXXXgrp24_CDE.rnnlm/complete_data/overlay.jpg"):
    # List to store converted images
    image_paths = os.listdir(image_dir)
    image_paths = [(image_dir + x) for x in image_paths if x.endswith("png")]
    images = []

    # Convert each image to mode 'L' and store in the images list
    for path in image_paths:
        img = Image.open(path).convert('L')
        images.append(img)

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    resized_images = [img.resize((max_width, max_height)) for img in images]

    # Create the initial composite with the first image
    composite = resized_images[0]

    # Overlay each subsequent image onto the composite
    alpha = 0.5
    for i in range(1, len(resized_images)):
        composite = Image.blend(composite, resized_images[i], alpha)

    composite.save(output_path)
    return composite

overlay_images("/scratches/dialfs/alta/bulats/import/reference_materials/scripts/vectra1/BULATS_Prompt_task_120813_RS_files_byScript/")