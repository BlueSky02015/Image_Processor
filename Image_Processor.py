import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import zipfile
import os

def brightness(image, brightness_value):
    img_array = np.array(image, dtype=np.int16)
    img_array = np.clip(img_array + brightness_value, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def to_grayscale(image):
    return ImageOps.grayscale(image)

def contrast(image, contrast_gain, contrast_pivot):
    img_array = np.array(image, dtype=np.int16)
    img_array = np.clip(contrast_gain * (img_array - contrast_pivot) + contrast_pivot, 0, 255)
    return Image.fromarray(img_array.astype(np.uint8))

def invert_colors(image):
    img_array = np.array(image)
    img_array = 255 - img_array
    return Image.fromarray(img_array)

def apply_kernel(image, kernel):
    img = np.asarray(image, dtype=np.float32)
    padded = np.pad(img, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.einsum("ijkl,kl->ij", windows, kernel)

def hpf0(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = apply_kernel(image, sobel_x)
    gy = apply_kernel(image, sobel_y)
    image_filtered = np.sqrt(gx ** 2 + gy ** 2)
    image_filtered = (image_filtered / image_filtered.max()) * 255
    return image_filtered.astype(np.uint8)

def hpf1(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 1, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 1, 0], [1, 2, 1]], dtype=np.float32)
    gx = apply_kernel(image, sobel_x)
    gy = apply_kernel(image, sobel_y)
    image_filtered = np.sqrt(gx ** 2 + gy ** 2)
    image_filtered = (image_filtered / image_filtered.max()) * 255
    return image_filtered.astype(np.uint8)

def tepi_titik(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    gx = apply_kernel(image, kernel)
    gy = apply_kernel(image, kernel)
    titik = np.abs(gx) + np.abs(gy)
    titik = (titik - titik.min()) / (titik.max() - titik.min()) * 255
    return Image.fromarray(titik.astype(np.uint8))

def tepi_garis(image):
    kernels = {
        "horizontal": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),
        "vertical": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),
        "diagonal_1": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32),
        "diagonal_2": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32),
    }
    img_array = np.asarray(image, dtype=np.float32)
    result = np.zeros_like(img_array)
    for kernel in kernels.values():
        temp_result = apply_kernel(img_array, kernel)
        result = np.maximum(result, temp_result)
    result = (result - result.min()) / (result.max() - result.min()) * 255
    return Image.fromarray(result.astype(np.uint8))

def tepi_arah(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    gx = apply_kernel(image, kernel)
    gy = apply_kernel(image, kernel)
    arah = np.arctan2(gy, gx)
    arah = (arah + np.pi) / (2 * np.pi) * 255
    return Image.fromarray(arah.astype(np.uint8))

def resize_aspect(image, ratio):
    if ratio == "Original":
        return image
    ratios = {
        "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4),
        "16:9": (16, 9), "9:16": (9, 16),
    }
    rw, rh = ratios[ratio]
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    target_ratio = rw / rh
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        image = image.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        image = image.crop((0, top, w, top + new_h))
    return image

# ----------------------------------------------------------------
# UI
# ----------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Image Processing")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    MAX_PROCESS_DIM = 1200
    THUMB_WIDTH = 350
    w, h = original_image.size

    scale = min(1.0, MAX_PROCESS_DIM / max(w, h))
    if scale < 1.0:
        image_resized = original_image.resize((int(w * scale), int(h * scale)))
    else:
        image_resized = original_image

    # Sidebar: Adjustments
    st.sidebar.header("🔧 Optimization Adjustments")
    brightness_value = st.sidebar.slider("Brightness", -100, 100, 0, step=1, key="brightness")
    brightness_image = brightness(image_resized, brightness_value)

    contrast_gain = st.sidebar.slider("Contrast Gain (G)", 1, 10, 3, step=1, key="contrast_gain")
    contrast_pivot = st.sidebar.slider("Contrast Pivot (P)", 0, 255, 128, step=1, key="contrast_pivot")
    contrast_image = contrast(image_resized, contrast_gain, contrast_pivot)

    invert = st.sidebar.checkbox("Invert Colors", value=True, key="invert")
    inverted_image = invert_colors(image_resized) if invert else image_resized

   
    # --- SEPARATOR ---
    st.sidebar.divider()

    # Generate derived images
    grayscale_image = to_grayscale(image_resized)
    grayscale_image_invert = to_grayscale(inverted_image)

    sketsa_image_hpf = hpf0(np.array(grayscale_image_invert))
    sketsa_image = invert_colors(sketsa_image_hpf)

    relief_image_hpf = hpf1(np.array(grayscale_image))

    tepi_image_titik = tepi_titik(grayscale_image)
    tepi_image_garis = tepi_garis(grayscale_image)
    tepi_image_arah = tepi_arah(grayscale_image)

    images = [
        {"name": "Original Image", "image": image_resized},
        {"name": "Contrast Image", "image": contrast_image},
        {"name": "Inverted Image", "image": inverted_image},
        {"name": "Brightness Image", "image": brightness_image},
        {"name": "Grayscale Image", "image": grayscale_image},
        {"name": "Grayscale Inverted Image", "image": grayscale_image_invert},
        {"name": "Sketsa Image (HPF0)", "image": sketsa_image_hpf},
        {"name": "Sketsa Image (Invert)", "image": sketsa_image},
        {"name": "Relief Image (HPF1)", "image": relief_image_hpf},
        {"name": "Tepi Titik Image", "image": tepi_image_titik},
        {"name": "Tepi Garis Image", "image": tepi_image_garis},
        {"name": "Tepi Arah Image", "image": tepi_image_arah},
    ]

    grid = []
    for i in range(0, len(images), 3):
        row = images[i:i + 3]
        grid.append(row)

    for row in grid:
        cols = st.columns(len(row))
        for i, img_item in enumerate(row):
            with cols[i]:
                st.image(img_item["image"], width=THUMB_WIDTH)
                st.write(img_item["name"])

    # ================= DOWNLOAD =====================
    st.sidebar.subheader("📥 Download Processed Images")
    aspect_ratio_choice = st.sidebar.selectbox(
        "Aspect Ratio",
        ["Original", "1:1", "4:3", "3:4", "16:9", "9:16"],
    )
    
    st.sidebar.markdown("### Select Images")
    selected_images = {}
    for item in images:
        selected_images[item["name"]] = st.sidebar.checkbox(
            item["name"],
            value=False,
            key=f"select_{item['name']}",
        )

    selected_count = sum(1 for v in selected_images.values() if v)
    original_filename = os.path.splitext(uploaded_file.name)[0]

    download_data = b""
    download_filename = ""
    download_mime = ""

    if selected_count == 1:
        selected_item = next(item for item in images if selected_images[item["name"]])
        img = selected_item["image"]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = resize_aspect(img, aspect_ratio_choice)
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        download_data = img_buffer.getvalue()
        
        version_name = selected_item["name"].replace(" ", "_")
        download_filename = f"{original_filename}_{version_name}.png"
        download_mime = "image/png"
        
    elif selected_count > 1:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item in images:
                if not selected_images[item["name"]]:
                    continue
                img = item["image"]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img = resize_aspect(img, aspect_ratio_choice)
                
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                
                version_name = item["name"].replace(" ", "_")
                filename = f"{original_filename}_{version_name}.png"
                zip_file.writestr(filename, img_buffer.getvalue())
                
        download_data = zip_buffer.getvalue()
        download_filename = f"{original_filename}_processed.zip"
        download_mime = "application/zip"

    st.sidebar.download_button(
        "📥 Download Image" if selected_count == 1 else "📦 Download Images",
        data=download_data,
        file_name=download_filename,
        mime=download_mime,
        disabled=(selected_count == 0),
    )

    if selected_count == 0:
        st.sidebar.caption("Select at least one image above to enable download.")

else:
    st.info("Upload an image to get started.")
# streamlit run Image_Processor.py
# tes