"""
Frag & Sieh — a picture helper for kids learning German.

A child asks a question in German (voice) -> Gemini writes down the German,
translates it to English, gives a simple German answer and 4 scene ideas ->
Nano Banana draws 4 child-friendly images.
"""

import io
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------- settings
TEXT_MODEL = "gemini-3.1-flash-lite"           # free tier: audio -> text + scenes
IMAGE_MODEL = "gemini-3.1-flash-lite-image"    # Nano Banana 2 Lite (needs billing)
N_IMAGES = 4
STYLE = ("children's book illustration, simple shapes, bright friendly colors, "
         "cute, no text, no letters")

st.set_page_config(page_title="Frag & Sieh", page_icon="🎈", layout="centered")


# ---------------------------------------------------------------- optional password
# If APP_PASSWORD is set in secrets, only people who know it can use the app.
# This protects your paid image credits from strangers who find the link.
def check_password():
    password = st.secrets.get("APP_PASSWORD")
    if not password or st.session_state.get("unlocked"):
        return True
    entered = st.text_input("🔒 Passwort", type="password")
    if entered == password:
        st.session_state["unlocked"] = True
        st.rerun()
    elif entered:
        st.error("Falsches Passwort")
    return False


# ---------------------------------------------------------------- models
@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


class KidQuestion(BaseModel):
    german: str            # what the child said, in German
    english: str           # English translation
    answer_de: str         # very simple answer in German (1-2 short sentences)
    safe: bool             # False if the question is not suitable for a kids app
    scenes: list[str]      # picture descriptions for the image model


INSTRUCTION = f"""
The audio is a young child (age 4-10) asking a question in German.
The child is learning German, so pronunciation and grammar may be imperfect.

1. german: write down what the child said, in correct German.
2. english: translate it into simple English.
3. answer_de: answer the question in 1-2 VERY short, simple German sentences a small child understands.
4. safe: false if the question is about violence, sex, drugs, self-harm or anything scary
   or unsuitable for small children; otherwise true.
5. scenes: {N_IMAGES} different picture descriptions (max 20 words each) that together help the
   child understand the answer. Describe ONLY what can be seen (objects, animals, people,
   actions, colors, places). No explanations, no science words, no written text in the pictures.
   Friendly and safe for small children. If safe is false, return an empty list.
"""


def understand_question(audio_bytes, mime_type):
    """One Gemini call: speech -> German text -> English -> answer -> scene ideas."""
    response = get_client().models.generate_content(
        model=TEXT_MODEL,
        contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type), INSTRUCTION],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=KidQuestion,
        ),
    )
    return response.parsed


def generate_one_image(prompt):
    try:
        response = get_client().models.generate_content(
            model=IMAGE_MODEL,
            contents=f"{prompt}, {STYLE}",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    except Exception as e:  # one failed image should not break the whole page
        print("Image error:", e)
    return None


def generate_images(prompts):
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(generate_one_image, prompts))


# ---------------------------------------------------------------- UI
st.markdown(
    """
    <style>
      .big-title {font-size: 2.6rem; font-weight: 800; text-align: center; margin-bottom: 0;}
      .subtitle  {text-align: center; font-size: 1.15rem; opacity: .75; margin-top: .2rem;}
      .bubble    {background: #FFF4D6; color: #3d2c00; border-radius: 18px;
                  padding: 14px 18px; font-size: 1.25rem; margin: 6px 0;}
      .answer    {background: #E3F4FF; color: #0b3550; border-radius: 18px;
                  padding: 14px 18px; font-size: 1.25rem; margin: 6px 0;}
      .small     {font-size: .95rem; opacity: .7;}
    </style>
    <p class="big-title">🎈 Frag &amp; Sieh</p>
    <p class="subtitle">Stell eine Frage auf Deutsch – ich zeige dir Bilder!</p>
    """,
    unsafe_allow_html=True,
)

if not check_password():
    st.stop()

audio = st.audio_input("🎙️ Drück auf das Mikrofon und frag mich etwas!")

if audio is not None:
    audio_bytes = audio.getvalue()
    key = hash(audio_bytes)

    # Only call the APIs once per recording (Streamlit reruns the script on every click)
    if st.session_state.get("last_key") != key:
        with st.spinner("👂 Ich höre zu …"):
            try:
                q = understand_question(audio_bytes, audio.type or "audio/wav")
            except Exception as e:
                st.error("Das hat leider nicht geklappt. Bitte versuch es noch einmal.")
                st.caption(f"Fehler: {e}")
                st.stop()

        images = []
        if q.safe and q.scenes:
            with st.spinner("🎨 Ich male Bilder für dich …"):
                images = generate_images(q.scenes[:N_IMAGES])

        st.session_state.update(last_key=key, result=q, images=images)

    q = st.session_state["result"]
    images = st.session_state["images"]

    st.markdown(f'<div class="bubble">🧒 {q.german}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="small">🇬🇧 {q.english}</p>', unsafe_allow_html=True)

    if not q.safe:
        st.markdown('<div class="answer">💛 Das ist eine gute Frage für einen Erwachsenen. '
                    'Frag doch Mama, Papa oder deine Lehrerin!</div>', unsafe_allow_html=True)
        st.stop()

    st.markdown(f'<div class="answer">💡 {q.answer_de}</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, img in enumerate(images):
        with cols[i % 2]:
            if img is not None:
                st.image(img, width="stretch")
            else:
                st.info("🖍️ Dieses Bild konnte ich nicht malen.")

    if images and all(img is None for img in images):
        st.warning("Keine Bilder – ist die Abrechnung (Billing) für den Gemini-Key aktiviert?")
