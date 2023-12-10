import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images_from_folder(folder, num_images=10, target_size=(224, 224)):
    images = []
    for filename in sorted(os.listdir(folder))[:num_images]:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            resized_img = cv2.resize(img, target_size)
            images.append(resized_img)
    return images


def align_images(images):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    base_image = images[0]
    gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    kp_base, des_base = sift.detectAndCompute(gray_base, None)

    aligned_images = [base_image]
    h, w = base_image.shape[:2]

    for image in images[1:]:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp_img, des_img = sift.detectAndCompute(gray_img, None)

        matches = bf.knnMatch(des_base, des_img, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) > 10:
            src_pts = np.float32([kp_base[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            aligned = cv2.warpPerspective(image, matrix, (w, h))
            aligned_images.append(aligned)
        else:
            aligned_images.append(image)

    return aligned_images


def plot_images_in_grid(images, rows=3, cols=3):
    plt.figure(figsize=(15, 15))
    for i, image in enumerate(images[:rows*cols]):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


folder = 'data/test/NORMAL'
images = load_images_from_folder(folder, num_images=50, target_size=(224, 224))
aligned_images = align_images(images)
plot_images_in_grid(aligned_images[:-9], rows=3, cols=3)
plot_images_in_grid(images, rows=3, cols=3)