import streamlit as st
from PIL import Image
import numpy as np
import io

# ----------------------------------------------------------------
# Region Sway Processor
# Standalone experimental app: click-and-drag a box over just the
# part of the image you want to move (e.g. the body, not the whole
# scene/background), and it sways rigidly left-right or up-down,
# feathered at the box edges so there's no visible seam. Everything
# outside the box stays perfectly still. No ML, no auto-segmentation
# needed — you tell it directly what should move.
#
# Requires an extra package for the click-and-drag rectangle:
#   pip install streamlit-drawable-canvas
# ----------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("🧍 Region Sway Processor (Box Select + Rigid Sway)")
st.caption(
    "Experimental. Draw a rectangle around just the part that should move "
    "(e.g. a character sitting inside a bigger scene). That region sways "
    "rigidly left-right or up-down, feathered at the edges. Everything "
    "outside the box stays still. Standalone testing file — upload any image."
)

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error(
        "This app needs an extra package for the click-and-drag box tool.\n\n"
        "Install it, then restart the app:\n\n"
        "```\npip install streamlit-drawable-canvas\n```"
    )
    st.stop()


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


def make_box_weight_mask(h, w, box, feather):
    """1.0 fully inside the box, smoothly fading to 0.0 within `feather` px
    outside the box edges, 0.0 further away."""
    x0, y0, x1, y1 = box
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    dx = np.maximum(np.maximum(x0 - x_idx, x_idx - x1), 0)
    dy = np.maximum(np.maximum(y0 - y_idx, y_idx - y1), 0)
    dist_outside = np.sqrt(dx.astype(np.float32) ** 2 + dy.astype(np.float32) ** 2)

    feather = max(feather, 1)
    weight = np.clip(1.0 - dist_outside / feather, 0.0, 1.0)
    return weight


def build_animation(image, weight_mask, num_frames, amplitude, axis, speed):
    img_array = np.array(image).astype(np.float32)
    h, w = weight_mask.shape
    y_idx, x_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    frames = []
    for i in range(num_frames):
        t = (i / num_frames) * 2 * np.pi  # one full sine cycle -> seamless loop
        d = amplitude * np.sin(speed * t)

        if axis == "Left-Right":
            dx_field = d * weight_mask
            dy_field = np.zeros_like(weight_mask)
        else:  # "Up-Down"
            dx_field = np.zeros_like(weight_mask)
            dy_field = d * weight_mask

        src_x = x_idx - dx_field
        src_y = y_idx - dy_field

        warped = bilinear_sample(img_array, src_x, src_y)
        frames.append(Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8), mode=image.mode))
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

    MAX_PROCESS_DIM = 900  # resolution used for the actual warp/export
    w, h = image.size
    scale = min(1.0, MAX_PROCESS_DIM / max(w, h))
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)))
    full_w, full_h = image.size

    # Smaller thumbnail just for drawing the box on (canvas widgets get
    # sluggish on very large images) — coordinates get rescaled back up.
    CANVAS_MAX_WIDTH = 700
    canvas_scale = min(1.0, CANVAS_MAX_WIDTH / full_w)
    canvas_w = int(full_w * canvas_scale)
    canvas_h = int(full_h * canvas_scale)
    canvas_bg = image.resize((canvas_w, canvas_h)).convert("RGB")

    st.subheader("1. Draw a box around the part that should sway")
    canvas_result = st_canvas(
        fill_color="rgba(255, 100, 0, 0.25)",
        stroke_width=2,
        stroke_color="#ff6400",
        background_image=canvas_bg,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        key="region_sway_canvas",
    )

    box = None
    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
        # Use the most recently drawn rectangle.
        obj = canvas_result.json_data["objects"][-1]
        rx = obj["left"]
        ry = obj["top"]
        rw = obj["width"] * obj.get("scaleX", 1)
        rh = obj["height"] * obj.get("scaleY", 1)

        # Rescale from canvas-space back to full processing resolution.
        x0 = rx / canvas_scale
        y0 = ry / canvas_scale
        x1 = (rx + rw) / canvas_scale
        y1 = (ry + rh) / canvas_scale
        box = (max(0, x0), max(0, y0), min(full_w, x1), min(full_h, y1))

    st.sidebar.header("🧍 Sway Settings")
    axis = st.sidebar.selectbox("Sway Direction", ["Left-Right", "Up-Down"])
    amplitude = st.sidebar.slider("Amplitude (px)", 0, 20, 4, step=1)
    feather = st.sidebar.slider("Edge Feather (px)", 5, 150, 40, step=5)
    speed = st.sidebar.slider("Speed", 1, 5, 1)
    num_frames = st.sidebar.slider("Frame Count", 6, 30, 12, step=1)
    frame_duration = st.sidebar.slider("Frame Duration (ms)", 20, 150, 65, step=5)
    st.sidebar.caption(f"Total loop length: ~{num_frames * frame_duration} ms")
    out_format = st.sidebar.selectbox(
        "Export Format", ["WEBP (keeps transparency)", "GIF (flattens transparency)"]
    )

    st.divider()
    st.subheader("2. Generate")

    if box is None:
        st.info("Draw a rectangle on the image above first (click and drag).")
    else:
        with st.expander("Preview weight mask (white = moves, black = still)"):
            preview_mask = make_box_weight_mask(full_h, full_w, box, feather)
            st.image((preview_mask * 255).astype(np.uint8), width=300)

        generate = st.button("🧍 Generate Sway Animation", type="primary")

        if generate:
            with st.spinner("Warping frames..."):
                weight_mask = make_box_weight_mask(full_h, full_w, box, feather)
                frames = build_animation(image, weight_mask, num_frames, amplitude, axis, speed)
                fmt = "WEBP" if out_format.startswith("WEBP") else "GIF"
                anim_bytes = frames_to_bytes(frames, fmt, frame_duration)

            st.subheader("Preview")
            st.image(anim_bytes)

            ext = "webp" if fmt == "WEBP" else "gif"
            st.download_button(
                f"📥 Download Animated {fmt}",
                data=anim_bytes,
                file_name=f"region_sway.{ext}",
                mime=f"image/{ext}",
            )
else:
    st.info(
        "Upload an image to try region sway. Tip: export a processed image "
        "from the Image Processor app first, then upload it here."
    )

# streamlit run region_sway_processor.py