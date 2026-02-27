# Streamlit App

import streamlit as st
import numpy as np
from PIL import Image
import joblib
from streamlit_drawable_canvas import st_canvas

modell = joblib.load("basta_modell.joblib")

st.set_page_config(page_title="MNIST Siffergissare", layout="centered")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
st.title("MNIST Siffergissare")
st.write("Rita en siffra (0–9) i rutan nedan, så gissar modellen vilken det är!")

canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    bild = canvas.image_data

    bild_pil = Image.fromarray(bild.astype("uint8"), "RGBA")
    bild_gra = bild_pil.convert("L")
    bild_array = np.array(bild_gra, dtype="float64")

    rader = np.any(bild_array > 0, axis=1)
    kolumner = np.any(bild_array > 0, axis=0)

    if rader.any():
        rmin, rmax = np.where(rader)[0][[0, -1]]
        kmin, kmax = np.where(kolumner)[0][[0, -1]]
        siffra = bild_array[rmin:rmax + 1, kmin:kmax + 1]
        siffra_pil = Image.fromarray(siffra)

        siffra_pil.thumbnail((20, 20), Image.LANCZOS)

        bild_28_array = np.zeros((28, 28), dtype="float64")
        y_off = (28 - siffra_pil.size[1]) // 2
        x_off = (28 - siffra_pil.size[0]) // 2
        bild_28_array[y_off:y_off + siffra_pil.size[1],
                      x_off:x_off + siffra_pil.size[0]] = np.array(siffra_pil)
    else:
        bild_28_array = np.zeros((28, 28), dtype="float64")

    bild_28 = Image.fromarray(bild_28_array.astype("uint8"))
    bild_array = bild_28_array

    st.subheader("Vad modellen ser (28x28):")
    st.image(bild_28, width=140)

    bild_flat = bild_array.reshape(1, 784)

    if bild_flat.sum() > 0:
        prediktion = modell.predict(bild_flat)[0]

        st.subheader("Modellens gissning:")
        st.markdown(
            f"<h1 style='text-align:center; font-size:120px;'>{prediktion}</h1>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Rita en siffra i rutan ovan!")
else:
    st.info("Rita en siffra i rutan ovan!")
