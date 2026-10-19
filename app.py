"""
Frag & Sieh — a talking picture friend for kids learning German.

Start page: the child picks one character friend and types their name.
Chat page: the child asks something in German by speaking. The friend
answers in very simple German and shows ONE picture of itself explaining it.
The friend's look stays the same in every picture because its portrait is
passed to Nano Banana as a reference image.
"""

import io

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------- settings
TEXT_MODEL = "gemini-3.1-flash-lite"           # free tier: listens, translates, answers
IMAGE_MODEL = "gemini-3.1-flash-lite-image"    # Nano Banana 2 Lite (needs billing)
HISTORY_TURNS = 6                              # how many past messages the friend remembers
STYLE = ("soft children's picture-book illustration, simple shapes, bright friendly colors, "
         "plain background, no text, no letters")

CHARACTERS = {
    "pia": {
        "name": "Pia", "emoji": "🐧", "color": "#3B82C4", "tint": "#DCEBFA",
        "tagline": "liebt Schnee, Fische und Rätsel",
        "look": "a small round penguin girl with a red knitted scarf and big friendly eyes",
        "personality": "curious and cheerful, loves snow, the sea and riddles",
    },
    "leo": {
        "name": "Leo", "emoji": "🦁", "color": "#E08A00", "tint": "#FFF0CC",
        "tagline": "ist mutig und mag Tiere",
        "look": "a small friendly lion cub with a fluffy orange mane and a green backpack",
        "personality": "brave and warm-hearted, loves animals, nature and adventures",
    },
    "robo": {
        "name": "Robo", "emoji": "🤖", "color": "#2A9D8F", "tint": "#D5F2EE",
        "tagline": "weiß viel über Technik und Sterne",
        "look": "a small round light-blue robot with a yellow antenna and a smiling screen face",
        "personality": "kind and a little funny, loves machines, space and numbers",
    },
    "fina": {
        "name": "Fina", "emoji": "🦊", "color": "#C2417A", "tint": "#FADDEA",
        "tagline": "malt gern und kennt den Wald",
        "look": "a little orange fox girl with a purple flower behind her ear and a white-tipped tail",
        "personality": "gentle and creative, loves painting, music, plants and the forest",
    },
}

st.set_page_config(page_title="Frag & Sieh", page_icon="🎈", layout="centered")

# ---------------------------------------------------------------- look & feel
card_css = "\n".join(
    f"""
    .st-key-card_{cid} {{
        background: {c['tint']}; border: 3px solid {c['color']} !important;
        border-radius: 26px; box-shadow: 6px 6px 0 {c['color']};
    }}
    .stApp .st-key-card_{cid} button {{ background: {c['color']}; border: none; }}
    .stApp .st-key-card_{cid} button p {{ color: #fff; }}
    """
    for cid, c in CHARACTERS.items()
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&display=swap');

    :root {{ --ink: #1F2A44; --paper: #F3FAF6; --sun: #FFC93C; }}
    .stApp {{ background: var(--paper); }}
    html, body, .stApp, .stApp p, .stApp button, .stApp input, .stApp label {{
        font-family: 'Baloo 2', system-ui, sans-serif; color: var(--ink);
    }}
    .hero {{ text-align: center; margin: .5rem 0 1.4rem; }}
    .hero h1 {{
        font-family: 'Baloo 2', sans-serif; font-weight: 800; font-size: 3.4rem;
        line-height: 1; margin: 0; color: var(--ink);
        text-shadow: 4px 4px 0 var(--sun);
    }}
    .hero p {{ font-size: 1.35rem; margin: .6rem 0 0; }}
    .stApp .card-emoji {{ font-size: 4.2rem; line-height: 1; text-align: center; margin-top: .4rem; }}
    .stApp .card-name  {{ font-size: 1.9rem; font-weight: 800; text-align: center; margin: .2rem 0 0; }}
    .stApp .card-tag   {{ font-size: 1.1rem; text-align: center; margin: 0 0 .4rem; min-height: 2.6em; }}
    .stButton button {{ border-radius: 999px; padding: .45rem 1rem; }}
    .stApp .stButton button p {{ font-weight: 700; font-size: 1.15rem; }}
    .stButton button:focus-visible {{ outline: 3px solid var(--ink); outline-offset: 2px; }}
    .stApp .st-key-start button {{ background: var(--sun); border: 3px solid var(--ink);
                                  box-shadow: 4px 4px 0 var(--ink); }}
    .stApp .st-key-start button p {{ font-size: 1.4rem; font-weight: 800; }}
    .stApp [data-testid="stWidgetLabel"] p {{ font-size: 1.2rem; font-weight: 700; }}
    .friend-head {{ display: flex; align-items: center; gap: .8rem; }}
    .friend-head h2 {{ margin: 0; font-size: 2rem; font-weight: 800; }}
    .en {{ font-size: .95rem; opacity: .65; margin-top: -.3rem; }}
    .stChatMessage p {{ font-size: 1.25rem; }}
    {card_css}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- optional password
def check_password():
    password = st.secrets.get("APP_PASSWORD")
    if not password or st.session_state.get("unlocked"):
        return True
    entered = st.text_input("🔒 Passwort", type="password")
    if entered == password:
        st.session_state["unlocked"] = True
        st.rerun()
    elif entered:
        st.error("Das Passwort stimmt nicht. Frag deine Eltern nach dem richtigen Passwort.")
    return False


# ---------------------------------------------------------------- Gemini
@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def image_from_response(response):
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    return None


def to_png_part(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")


@st.cache_resource(show_spinner=False)
def get_portrait(char_id):
    """Draw each friend once per server start and share it with all users.
    The portrait is the reference that keeps the friend looking the same."""
    c = CHARACTERS[char_id]
    try:
        response = get_client().models.generate_content(
            model=IMAGE_MODEL,
            contents=f"Character portrait: {c['look']}, waving hello, full body, {STYLE}",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
            ),
        )
        return image_from_response(response)
    except Exception as e:
        print("Portrait error:", e)
        return None


def draw_scene(char_id, scene):
    """One picture of the friend showing the answer, matching the portrait."""
    c = CHARACTERS[char_id]
    portrait = get_portrait(char_id)
    prompt = (f"Draw this exact same character ({c['look']}) in a new scene: {scene}. "
              f"Keep the character's look identical to the reference. {STYLE}")
    contents = [to_png_part(portrait), prompt] if portrait else prompt
    try:
        response = get_client().models.generate_content(
            model=IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="4:3"),
            ),
        )
        return image_from_response(response)
    except Exception as e:
        print("Scene error:", e)
        return None


class Turn(BaseModel):
    child_de: str      # what the child said, in correct German
    child_en: str      # English translation (for parents)
    reply_de: str      # the friend's answer in very simple German
    reply_en: str      # English translation of the answer
    safe: bool         # False if the topic is not suitable for small children
    scene: str         # one picture: the friend showing the answer


def build_instruction(char_id, child_name, history):
    c = CHARACTERS[char_id]
    past = "\n".join(
        f"{'Child' if m['role'] == 'user' else c['name']}: {m['de']}"
        for m in history[-HISTORY_TURNS:]
    ) or "(no messages yet)"
    return f"""
You are {c['name']}, {c['look']}. You are {c['personality']}.
You are the friend of a child called {child_name or 'the child'} (age 4-10) who lives in Germany
and is learning German. Their German may be imperfect.

Conversation so far:
{past}

The audio is the child speaking German.

Fill in:
- child_de: what the child said, written in correct German.
- child_en: English translation of that.
- reply_de: your answer as {c['name']}, in VERY simple German (level A1): max 3 short sentences,
  common words, present tense, warm and encouraging. Use the child's name sometimes.
  End with one easy question so the conversation continues.
  If the child's German had a mistake, simply use the correct words in your answer, never scold.
- reply_en: English translation of reply_de.
- safe: false if the child asks about violence, sex, drugs, self-harm or other topics not suitable
  for small children; then reply_de kindly says this is a good question for a grown-up
  (Mama, Papa or a teacher) and suggests something else to talk about.
- scene: ONE picture description (max 30 words) showing {c['name']} doing something that shows
  your answer. Describe only what can be seen. No text in the picture. Friendly and safe.
"""


def ask_friend(char_id, child_name, history, audio):
    instruction = build_instruction(char_id, child_name, history)
    contents = [types.Part.from_bytes(data=audio.getvalue(), mime_type=audio.type or "audio/wav"),
                instruction]
    response = get_client().models.generate_content(
        model=TEXT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=Turn),
    )
    return response.parsed


# ---------------------------------------------------------------- pages
def start_page():
    st.markdown(
        '<div class="hero"><h1>Frag &amp; Sieh</h1>'
        "<p>Such dir einen Freund aus. Er erklärt dir die Welt – auf Deutsch und mit Bildern.</p></div>",
        unsafe_allow_html=True,
    )

    chosen = st.session_state.get("pick")
    ids = list(CHARACTERS)
    for row in (ids[:2], ids[2:]):
        cols = st.columns(2, gap="medium")
        for col, cid in zip(cols, row):
            c = CHARACTERS[cid]
            with col, st.container(key=f"card_{cid}", border=True):
                st.markdown(
                    f'<div class="card-emoji">{c["emoji"]}</div>'
                    f'<p class="card-name">{c["name"]}</p>'
                    f'<p class="card-tag">{c["tagline"]}</p>',
                    unsafe_allow_html=True,
                )
                label = f"✔ {c['name']} ist dabei" if chosen == cid else f"{c['name']} wählen"
                if st.button(label, key=f"pick_{cid}", width="stretch"):
                    st.session_state["pick"] = cid
                    st.rerun()

    st.write("")
    name = st.text_input("Wie heißt du?", key="child_name_input", placeholder="Dein Name")

    if chosen:
        if st.button(f"Mit {CHARACTERS[chosen]['name']} sprechen", key="start", width="stretch"):
            st.session_state.update(page="chat", char=chosen, child_name=name.strip(),
                                    messages=[], turn=0)
            st.rerun()
    else:
        st.caption("Tipp auf einen Freund, dann geht es los.")


def chat_page():
    cid = st.session_state["char"]
    c = CHARACTERS[cid]
    child_name = st.session_state.get("child_name", "")
    messages = st.session_state["messages"]

    head_l, head_r = st.columns([3, 2], vertical_alignment="center")
    with head_l:
        st.markdown(f'<div class="friend-head"><h2>{c["emoji"]} {c["name"]}</h2></div>',
                    unsafe_allow_html=True)
    with head_r:
        if st.button("Anderen Freund wählen", key="back"):
            st.session_state.update(page="start", pick=None)
            st.rerun()

    # Greeting with the friend's portrait (drawn once, then cached for everyone)
    if not messages:
        with st.spinner(f"{c['name']} kommt gleich …"):
            portrait = get_portrait(cid)
        hello = f"Hallo{' ' + child_name if child_name else ''}! Ich bin {c['name']}. " \
                f"Frag mich etwas – ich zeige es dir!"
        messages.append({"role": "friend", "de": hello,
                         "en": f"Hi! I'm {c['name']}. Ask me something – I'll show you!",
                         "image": portrait})

    for m in messages:
        avatar = c["emoji"] if m["role"] == "friend" else "🧒"
        with st.chat_message("assistant" if m["role"] == "friend" else "user", avatar=avatar):
            st.markdown(m["de"])
            if m.get("en"):
                st.markdown(f'<p class="en">🇬🇧 {m["en"]}</p>', unsafe_allow_html=True)
            if m.get("image") is not None:
                st.image(m["image"], width=380)

    # New widget key every turn, so the recorder is empty again after each question
    turn = st.session_state["turn"]
    audio = st.audio_input("🎙️ Drück auf das Mikrofon und sprich", key=f"mic_{turn}")

    if audio is None:
        return

    with st.spinner(f"{c['name']} hört zu …"):
        try:
            reply = ask_friend(cid, child_name, messages, audio)
        except Exception as e:
            st.error(f"{c['name']} hat dich nicht verstanden. Sprich bitte noch einmal.")
            st.caption(f"Technischer Fehler: {e}")
            st.session_state["turn"] = turn + 1
            return

    image = None
    if reply.safe and reply.scene:
        with st.spinner(f"{c['name']} malt ein Bild für dich …"):
            image = draw_scene(cid, reply.scene)

    messages.append({"role": "user", "de": reply.child_de, "en": reply.child_en})
    messages.append({"role": "friend", "de": reply.reply_de, "en": reply.reply_en, "image": image})
    st.session_state["turn"] = turn + 1
    st.rerun()


# ---------------------------------------------------------------- router
if check_password():
    if st.session_state.get("page") == "chat":
        chat_page()
    else:
        start_page()
