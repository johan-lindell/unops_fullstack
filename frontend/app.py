import sys
import os

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXAMPLE_TWEETS = [
    "Forest fire near La Ronge Sask. Canada",
    "I love sunsets on the beach, so peaceful today!",
    "13,000 people receive #wildfires evacuation orders in California",
    "Just got a new puppy, he's so cute!",
]

st.set_page_config(
    page_title="Disaster Tweet Classifier",
    page_icon="🚨",
    layout="centered",
)

st.title("🚨 Disaster Tweet Classifier")
st.caption(
    "Paste a tweet below to find out whether it describes a real disaster or not."
)

with st.sidebar:
    st.header("About the model")
    try:
        from model.predict import MODEL_INFO

        st.metric("Algorithm", MODEL_INFO.get("algorithm", "—"))
        st.metric("Training F1", f"{MODEL_INFO.get('f1', 0):.3f}")
        st.metric("Training samples", f"{MODEL_INFO.get('n_train', 0):,}")
    except Exception:
        st.info("Model metadata unavailable. Train the model first.")

    st.divider()
    st.markdown(
        "**How to interpret confidence:**\n"
        "- > 80%: high confidence\n"
        "- 60–80%: moderate confidence\n"
        "- < 60%: uncertain — treat with caution"
    )

col_predict, col_example = st.columns([2, 1])

with col_example:
    if st.button("Load example", use_container_width=True):
        import random
        st.session_state["tweet_input"] = random.choice(EXAMPLE_TWEETS)

tweet_text = st.text_area(
    "Tweet text",
    height=120,
    placeholder="e.g. Forest fire near La Ronge Sask. Canada",
    help="Paste any short text and the model will classify it.",
    key="tweet_input",
)

with col_predict:
    predict_clicked = st.button("Analyze", type="primary", use_container_width=True)

if predict_clicked:
    if not tweet_text or not tweet_text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            from model.predict import predict_single
        except ImportError:
            st.error(
                "Model module not found. Make sure `model/predict.py` exists "
                "and the model artifact has been trained (`python model/train.py`)."
            )
            st.stop()

        with st.spinner("Analyzing…"):
            try:
                result = predict_single(tweet_text.strip())
            except FileNotFoundError as exc:
                st.error(
                    f"Model artifact missing: {exc}\n\n"
                    "Run `python model/train.py` to train the model first."
                )
                st.stop()
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        label = result["label"]
        label_name = result["label_name"]
        confidence = result["confidence"]

        if label == 1:
            st.markdown(
                f'<div style="background:#ff4b4b;color:white;padding:16px 20px;'
                f'border-radius:8px;font-size:1.3rem;font-weight:700;">'
                f"🔴 {label_name}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#21c354;color:white;padding:16px 20px;'
                f'border-radius:8px;font-size:1.3rem;font-weight:700;">'
                f"🟢 {label_name}</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        st.write(f"**Confidence:** {confidence:.1%}")
        st.progress(confidence)

        direction = "a real disaster" if label == 1 else "not a real disaster"
        certainty = (
            "high confidence"
            if confidence > 0.8
            else "moderate confidence"
            if confidence > 0.6
            else "low confidence — treat with caution"
        )
        st.caption(
            f"The model is **{confidence:.0%}** confident this tweet describes "
            f"**{direction}** ({certainty})."
        )
