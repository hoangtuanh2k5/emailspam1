import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ===== Load model =====
model = joblib.load("phanloaiemail.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ===== Page Config =====
st.set_page_config(
    page_title="Spam Email Classifier Pro",
    page_icon="!",
    layout="wide",
)

# ===== Theme toggle =====
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if theme == "Light":
    colors = {
        "bg": "#ffffff",
        "card": "#ffffff",
        "border": "#e5e7eb",
        "text": "#111827",
        "muted": "#6b7280",
        "accent": "#2563eb",
        "accent2": "#4f46e5",
        "hero1": "#2563eb",
        "hero2": "#1d4ed8",
        "hero_text": "#f8fafc",
    }
else:
    colors = {
        "bg": "#0b1220",
        "card": "#111827",
        "border": "#1f2937",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "accent": "#60a5fa",
        "accent2": "#7c3aed",
        "hero1": "#0f172a",
        "hero2": "#111827",
        "hero_text": "#e5e7eb",
    }

st.markdown(
    """
    <style>
    :root {
        --bg: %(bg)s;
        --card: %(card)s;
        --border: %(border)s;
        --text: %(text)s;
        --text-muted: %(muted)s;
        --accent: %(accent)s;
        --accent-2: %(accent2)s;
        --hero-text: %(hero_text)s;
    }
    body { background: var(--bg); color: var(--text); }
    .main { padding-top: 1rem; }
    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, %(hero1)s, %(hero2)s);
        color: var(--hero-text);
        border: 1px solid var(--border);
        box-shadow: 0 6px 20px rgba(0,0,0,0.10);
        margin-bottom: 1rem;
    }
    .card {
        padding: 1rem;
        border-radius: 12px;
        background: var(--card);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        color: var(--text);
    }
    .metric { font-size: 28px; font-weight: 700; }
    .subtle { color: var(--text-muted); }
    .stButton button {
        background: linear-gradient(120deg, var(--accent-2), var(--accent));
        color: white;
        border: none;
        padding: 0.5rem 1.2rem;
        border-radius: 8px;
    }
    .stDownloadButton button {
        background: linear-gradient(120deg,var(--accent-2),var(--accent));
        color: white;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        background: var(--card);
        border: 1px solid var(--border);
        color: var(--text);
    }
    .stDataFrame, .stTable { background: var(--card); color: var(--text); }
    [data-testid="stMetricValue"] { color: var(--text); }
    </style>
    """ % colors,
    unsafe_allow_html=True,
)


# ===== Helpers =====
def card(title: str, desc: str, icon: str = "*", link: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="card">
            <h3 style="margin:0;">{icon} {title}</h3>
            <p class="subtle" style="margin:0.3rem 0 0.4rem 0;">{desc}</p>
            {f'<a href="{link}" target="_blank" style="text-decoration:none; color:#2563eb; font-weight:600;">Visit</a>' if link else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_labels(model, vectorizer, texts):
    """Return predicted labels."""
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    return y_pred


# ===== Header =====
st.markdown(
    "<div class='hero'>"
    "<h1 style='text-align:center; margin:0;'>Spam Email Classifier</h1>"
    "<p style='text-align:center; color:var(--hero-text); margin:0.4rem 0 0 0;'>TF-IDF + multiple models, saved best model for inference.</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["Dashboard", "Test Email", "Batch Upload"])


# ===== Dashboard =====
with tab1:
    st.info("This app classifies Spam/Ham using a best-performing model trained with TF-IDF features.")

    col1, col2, col3 = st.columns(3)
    with col1:
        card("Model", "Best model from training")
    with col2:
        card("Vectorizer", "TF-IDF")
    with col3:
        card("Metrics", "See evaluation in notebook")

    with st.expander("Sample Spam/Ham distribution"):
        sample = pd.DataFrame({"Type": ["Spam", "Ham"], "Count": [60, 40]})
        fig_d, ax_d = plt.subplots(figsize=(4, 3))
        sns.barplot(data=sample, x="Type", y="Count", palette="coolwarm", ax=ax_d)
        ax_d.set_xlabel("")
        ax_d.set_ylabel("Count")
        st.pyplot(fig_d, use_container_width=False)


# ===== Test Email =====
with tab2:
    st.subheader("Test Email (Realtime)")

    review = st.text_area("Enter email content:", height=150)
    if st.button("Classify"):
        if review.strip():
            y_pred_arr = predict_labels(model, vectorizer, [review])
            y_pred = y_pred_arr[0]

            st.write("### Result:")
            if y_pred == 1:
                st.error("Spam Detected!")
            else:
                st.success("Safe (Ham)")

            # highlight keyword
            if y_pred == 1:
                keywords = ["free", "click", "win", "offer"]
                highlighted = review
                for k in keywords:
                    highlighted = highlighted.replace(
                        k, f"<mark style='background:red'>{k}</mark>"
                    )
                st.markdown(f"### Highlighted Email\n{highlighted}", unsafe_allow_html=True)
        else:
            st.warning("Please enter email content!")


# ===== Batch Upload =====
with tab3:
    with st.expander("Upload CSV file"):
        file_upload = st.file_uploader("Choose CSV file", type=["csv"])
    if file_upload is not None:
        data = pd.read_csv(file_upload).dropna().drop_duplicates()

        data = data[data["Category"].isin(["ham", "spam"])]  # Keep only spam/ham samples
        if "Message" not in data.columns:
            st.error("File must contain a `Message` column")
        else:
            y_pred = predict_labels(model, vectorizer, data["Message"])

            data["Prediction"] = ["Spam" if p == 1 else "Ham" for p in y_pred]

            st.success("Classification complete!")
            with st.expander("View detailed predictions"):
                c1, c2, c3 = st.columns([1, 3, 1])
                with c2:
                    st.dataframe(data[["Message", "Prediction"]], use_container_width=True)

            y_test = data["Category"].map({"ham": 0, "spam": 1})
            cm = confusion_matrix(y_test, y_pred)

            with st.expander("Model evaluation"):
                col_A, _, col_B = st.columns([5, 1, 5])
                with col_A:
                    fig_, ax_ = plt.subplots()
                    sns.countplot(data=data, x="Prediction", hue="Prediction", palette="coolwarm", ax=ax_)
                    ax_.legend(title="Email Type", labels=["Ham", "Spam"], loc="upper right", fontsize=10, frameon=True)
                    ax_.set_title("Spam/Ham Distribution")
                    st.pyplot(fig_, use_container_width=True)

                with col_B:
                    fig, ax = plt.subplots()
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Ham", "Spam"],
                        yticklabels=["Ham", "Spam"],
                        ax=ax,
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig, use_container_width=True)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results",
                csv,
                "spam_predictions.csv",
                "text/csv",
                key="download-csv",
            )
