import streamlit as st
from PIL import Image
import numpy as np
import io

# ----------------------------------------------------------------
# Weighted Wave Processor (Tier 2)
# Standalone experimental app: one side of the image stays anchored
# / still, and sway motion increases toward the opposite side —
# similar to how hanging hair or a skirt hem behaves. Separate from
# jelly_wave_processor.py (Tier 1) and Image_Processor.py on purpose,
# so this effect can be tested/tuned in isolation.
# ----------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🎐 Weighted Wave Processor (Tier 2 — Region-Weighted Sway)")
st.caption(
    "Experimental. One edge of the image stays anchored/still, motion increases "
    "toward the opposite edge — like hair or a skirt swaying. Standalone testing file — "
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


def make_weight_mask(h, w, anchor, falloff):
    """0 near the anchored edge, ramping toward 1 at the opposite edge.
    `falloff` > 1 keeps the anchor area stiffer for longer before swaying kicks in;
    `falloff` < 1 spreads the motion in sooner."""
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    if anchor == "Top (sways downward)":
        weight = y_idx / max(h - 1, 1)
    elif anchor == "Bottom (sways upward)":
        weight = 1 - (y_idx / max(h - 1, 1))
    elif anchor == "Left (sways rightward)":
        weight = x_idx / max(w - 1, 1)
    else:  # "Right (sways leftward)"
        weight = 1 - (x_idx / max(w - 1, 1))

    return weight ** falloff


def weighted_wave_frame(img_array, t, weight_mask, amp_x, amp_y, wavelength, speed):
    h, w = img_array.shape[:2]
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    phase = speed * t
    dx = amp_x * np.sin(2 * np.pi * y_idx / max(wavelength, 1) + phase) * weight_mask
    dy = amp_y * np.sin(2 * np.pi * x_idx / max(wavelength, 1) + phase * 0.85) * weight_mask

    src_x = x_idx - dx
    src_y = y_idx - dy

    warped = bilinear_sample(img_array.astype(np.float32), src_x, src_y)
    return np.clip(warped, 0, 255).astype(np.uint8)


def build_animation(image, num_frames, weight_mask, amp_x, amp_y, wavelength, speed):
    img_array = np.array(image)
    frames = []
    for i in range(num_frames):
        t = (i / num_frames) * 2 * np.pi  # one full sine cycle -> seamless loop
        frame = weighted_wave_frame(img_array, t, weight_mask, amp_x, amp_y, wavelength, speed)
        frames.append(Image.fromarray(frame, mode=image.mode))
    return frames


def frames_to_bytes(frames, fmt, duration_ms):
    buffer = io.BytesIO()
    if fmt == "GIF":
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

    st.sidebar.header("🎐 Sway Settings")
    anchor = st.sidebar.selectbox(
        "Anchor Side (stays still)",
        ["Top (sways downward)", "Bottom (sways upward)", "Left (sways rightward)", "Right (sways leftward)"],
    )
    falloff = st.sidebar.slider(
        "Anchor Stiffness", 0.5, 4.0, 2.0, step=0.1,
        help="Higher = anchor area stays stiffer longer before swaying kicks in",
    )
    amp_x = st.sidebar.slider("Horizontal Amplitude (px)", 0, 40, 10)
    amp_y = st.sidebar.slider("Vertical Amplitude (px)", 0, 40, 4)
    wavelength = st.sidebar.slider("Wavelength (px)", 20, 400, 150)
    speed = st.sidebar.slider("Speed", 1, 10, 3)
    num_frames = st.sidebar.slider("Frame Count", 8, 40, 20)
    frame_duration = st.sidebar.slider("Frame Duration (ms)", 20, 200, 60)
    out_format = st.sidebar.selectbox(
        "Export Format", ["WEBP (keeps transparency)", "GIF (flattens transparency)"]
    )

    weight_mask = make_weight_mask(image.size[1], image.size[0], anchor, falloff)

    with st.expander("Preview weight mask (white = moves more, black = anchored)"):
        st.image((weight_mask * 255).astype(np.uint8), width=300)

    st.divider()
    generate = st.button("🎐 Generate Sway Animation", type="primary")

    if generate:
        with st.spinner("Warping frames..."):
            frames = build_animation(image, num_frames, weight_mask, amp_x, amp_y, wavelength, speed)
            fmt = "WEBP" if out_format.startswith("WEBP") else "GIF"
            anim_bytes = frames_to_bytes(frames, fmt, frame_duration)

        st.subheader("Preview")
        st.image(anim_bytes)

        ext = "webp" if fmt == "WEBP" else "gif"
        st.download_button(
            f"📥 Download Animated {fmt}",
            data=anim_bytes,
            file_name=f"weighted_wave.{ext}",
            mime=f"image/{ext}",
        )
    else:
        st.info("Adjust the sliders on the left, then click **Generate Sway Animation** above.")
else:
    st.info(
        "Upload an image to try the weighted sway effect. Tip: export a processed "
        "image from the Image Processor app first, then upload it here."
    )

# streamlit run weighted_wave_processor.py