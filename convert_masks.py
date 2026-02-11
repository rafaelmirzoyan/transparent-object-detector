import os
import cv2
import numpy as np

#Root folder for masks & images
ROOT = r"C:\Users\Peter\Downloads\train\train"

img_dir = os.path.join(ROOT, "images")
mask_dir = os.path.join(ROOT, "masks")

# Output for the YOLO-converted dataset
OUT_ROOT = os.path.join(ROOT, "yolo_out")
out_img_dir = os.path.join(OUT_ROOT, "images")
out_lbl_dir = os.path.join(OUT_ROOT, "labels")

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

#These print where the locations of where they're going
print("Image dir:", img_dir)
print("Mask dir :", mask_dir)
print("Output images:", out_img_dir)
print("Output labels:", out_lbl_dir)

# Image extensions that are usable here, no other are allowed just in case
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

count_imgs = 0
count_with_masks = 0

for img_name in os.listdir(img_dir):
    if not img_name.lower().endswith(IMAGE_EXTS):
        continue
    #Increments the number of images
    count_imgs += 1

    img_path = os.path.join(img_dir, img_name)
    base = os.path.splitext(img_name)[0]

    #All masks are labeled with the same format, number_mask.png so that makes it easy
    mask_name = base + "_mask.png"
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"[WARN] No mask file for {img_name} (expected {mask_name})")
        continue

    #Copy image to YOLO image folder
    out_img_path = os.path.join(out_img_dir, img_name)
    if not os.path.exists(out_img_path):
        with open(img_path, "rb") as src, open(out_img_path, "wb") as dst:
            dst.write(src.read())

    #Reads pixel mask
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[WARN] Could not read mask {mask_path}") #Error message for pixel mask
        continue

    #CXonvert to gray-scale
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    #Creates binary mask because background, 255 = object
    binary = (gray > 0).astype(np.uint8) * 255

    # Find contours of the object(s)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[WARN] No contours found in mask {mask_name}")
        continue

    #Find dimensions of the image (Thank you Cv2!)
    img = cv2.imread(img_path) #returns the matrix
    if img is None:
        print(f"[WARN] Could not read image {img_path}")
        continue
    h, w = img.shape[:2]

    label_path = os.path.join(out_lbl_dir, base + ".txt")
    with open(label_path, "w") as f:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30: #Ignores small error specks and stuff
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            bw_n = bw / w
            bh_n = bh / h
            #Stuff for normalization
            cnt = cnt.reshape(-1, 2)
            xs = cnt[:, 0] / w
            ys = cnt[:, 1] / h

            #class id 0 = transparent_object
            f.write(f"0 {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
            for px, py in zip(xs, ys):
                f.write(f" {px:.6f} {py:.6f}")
            f.write("\n")

    count_with_masks += 1
#Some cool messages for final stuff
print(f"\nDone. Processed {count_imgs} images, wrote labels for {count_with_masks} with masks.")
print("YOLO-style dataset is in:", OUT_ROOT)
#This sucked a lot. We spent too long on this.