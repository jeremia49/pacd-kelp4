import streamlit as st
from PIL import Image
from process import ImageProcessor
import numpy as np
from preprocess import preprocess

@st.cache_resource
def load_model():
    return ImageProcessor()

st.set_page_config(page_title="Klasifikasi TBC Paru-Paru", layout="centered")

st.markdown(
    """
    # 🫁 Klasifikasi TBC Paru-Paru  
    ### Kelompok 4 - **PACD**
    **Anggota:**  
    - Jeremia Manurung 25/563379/PPA/07110
    - Rifda Sasmi Zahra 25/567016/PPA/07142
    - Theofilus Arkhi Susanto 25/555265/PPA/07030

    ---
    Upload gambar rontgen paru-paru untuk mengetahui apakah terindikasi **TBC** atau **tidak**.  
    Pastikan gambar dalam format **JPG / PNG**.
    """
)

uploaded_file = st.file_uploader(
    "📤 Upload gambar rontgen paru-paru di sini:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    st.write("")  # spasi
    
    
    if isinstance(image, Image.Image):
        image = np.array(image)

    if st.button("🔍 Klasifikasikan"):
        st.write("Memproses gambar... Mohon tunggu sebentar.")

        model = load_model()
        label = ''
        try:
            processedimage = preprocess(image)
            segmentedimage = model.segment(processedimage)
            result = model.process(segmentedimage)
            if(result) == 'tb':
                label = "TBC"
            else:
                label = "Normal"
        finally:
            pass

        # Tampilkan hasil akhir
        st.subheader("🩺 Hasil Klasifikasi")
        if label == "TBC":
            st.error("Terindikasi **TBC Paru-Paru** 😷")
        elif label == "Normal":
            st.success("Tidak Terindikasi **TBC Paru-Paru** ✅")
        else:
            st.error("Sistem Error")


else:
    st.info("Silakan upload gambar terlebih dahulu untuk memulai klasifikasi.")