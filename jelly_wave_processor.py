import streamlit as st
from PIL import Image
import numpy as np
import io

# ----------------------------------------------------------------
# Jelly Wave Processor (Tier 1)
# Standalone experimental app: applies a UNIFORM sine-wave "jelly"
# wobble to the whole image (every pixel warps the same way based
# on its position + time). Not part of the main Image_Processor.py
# on purpose, so this effect can be tested/tuned in isolation.
# ----------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🌊 Jelly Wave Processor (Tier 1 — Whole-Image Warp)")
st.caption(
    "Experimental. Applies a uniform sine-wave wobble to the whole image "
    "and exports it as a short looping animation. Standalone testing file — "
    "upload any image (e.g. one exported from the Image Processor) to try it."
)


def bilinear_sample(img, src_x, src_y):
    """Sample img at fractional (src_x, src_y) coordinates using bilinear interpolation.
    Pure numpy, no scipy dependency."""
    h, w = img.shape[:2]
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = src_x - x0
    wy = src_y - y0

    if img.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    top = Ia * (1 - wx) + Ib * wx
    bottom = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bottom * wy


def jelly_warp_frame(img_array, t, amp_x, amp_y, wavelength_x, wavelength_y, speed):
    h, w = img_array.shape[:2]
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    phase = speed * t
    dx = amp_x * np.sin(2 * np.pi * y_idx / max(wavelength_y, 1) + phase)
    dy = amp_y * np.sin(2 * np.pi * x_idx / max(wavelength_x, 1) + phase * 0.85)

    src_x = x_idx - dx
    src_y = y_idx - dy

    warped = bilinear_sample(img_array.astype(np.float32), src_x, src_y)
    return np.clip(warped, 0, 255).astype(np.uint8)


def build_animation(image, num_frames, amp_x, amp_y, wavelength_x, wavelength_y, speed):
    img_array = np.array(image)
    frames = []
    for i in range(num_frames):
        t = (i / num_frames) * 2 * np.pi  # one full sine cycle -> seamless loop
        frame = jelly_warp_frame(img_array, t, amp_x, amp_y, wavelength_x, wavelength_y, speed)
        frames.append(Image.fromarray(frame, mode=image.mode))
    return frames


def frames_to_bytes(frames, fmt, duration_ms):
    buffer = io.BytesIO()
    if fmt == "GIF":
        # GIF has no real alpha blending, so flatten transparency onto white first.
        gif_frames = []
        for f in frames:
            if f.mode == "RGBA":
                bg = Image.new("RGB", f.size, (255, 255, 255))
                bg.paste(f, mask=f.split()[3])
                gif_frames.append(bg.convert("P", palette=Image.ADAPTIVE))
            else:
                gif_frames.append(f.convert("P", palette=Image.ADAPTIVE))
        gif_frames[0].save(
            buffer, format="GIF", save_all=True, append_images=gif_frames[1:],
            duration=duration_ms, loop=0, disposal=2,
        )
    else:  # WEBP keeps full alpha transparency
        frames[0].save(
            buffer, format="WEBP", save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0,
        )
    return buffer.getvalue()


uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    image = image.convert("RGBA") if has_alpha else image.convert("RGB")

    MAX_DIM = 800  # keep animation generation snappy
    w, h = image.size
    scale = min(1.0, MAX_DIM / max(w, h))
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)))

    st.image(image, caption="Uploaded image", width=300)

    st.sidebar.header("🌊 Wave Settings")
    amp_x = st.sidebar.slider("Horizontal Amplitude (px)", 0, 40, 8)
    amp_y = st.sidebar.slider("Vertical Amplitude (px)", 0, 40, 5)
    wavelength_x = st.sidebar.slider("Wavelength X (px)", 20, 400, 150)
    wavelength_y = st.sidebar.slider("Wavelength Y (px)", 20, 400, 150)
    speed = st.sidebar.slider("Speed", 1, 10, 3)
    num_frames = st.sidebar.slider("Frame Count", 8, 40, 20)
    frame_duration = st.sidebar.slider("Frame Duration (ms)", 20, 200, 60)
    out_format = st.sidebar.selectbox(
        "Export Format", ["WEBP (keeps transparency)", "GIF (flattens transparency)"]
    )

    st.divider()
    generate = st.button("🌊 Generate Wave Animation", type="primary")

    if generate:
        with st.spinner("Warping frames..."):
            frames = build_animation(
                image, num_frames, amp_x, amp_y, wavelength_x, wavelength_y, speed
            )
            fmt = "WEBP" if out_format.startswith("WEBP") else "GIF"
            anim_bytes = frames_to_bytes(frames, fmt, frame_duration)

        st.subheader("Preview")
        st.image(anim_bytes)

        ext = "webp" if fmt == "WEBP" else "gif"
        st.download_button(
            f"📥 Download Animated {fmt}",
            data=anim_bytes,
            file_name=f"jelly_wave.{ext}",
            mime=f"image/{ext}",
        )
    else:
        st.info("Adjust the sliders on the left, then click **Generate Wave Animation** above.")
else:
    st.info(
        "Upload an image to try the jelly wave effect. Tip: export a processed "
        "image from the Image Processor app first, then upload it here."
    )

# streamlit run jelly_wave_processor.py