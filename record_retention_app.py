import streamlit as st
import pandas as pd
import pdfplumber
import pytesseract
import docx2txt
import fitz  # pymupdf
import os
import tempfile
import datetime
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import zipfile

st.set_page_config(page_title="📁 Washington Records Retention Assistant")
st.title("📁 Washington Records Retention Assistant")


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# Extract retention rules from PDF
def extract_retention_from_pdf(pdf_file):
    rules = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table[1:]:
                    if row and len(row) >= 3:
                        category = row[0] if row[0] else ""
                        description = row[1] if row[1] else ""
                        retention = row[2] if row[2] else ""
                        rules.append({
                            "category_name": category.strip(),
                            "category_description": description.strip(),
                            "retention_period": retention.strip()
                        })
    return pd.DataFrame(rules)


# Extract text from various file types
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif uploaded_file.name.endswith(".docx"):
        text = docx2txt.process(uploaded_file)
        if text.strip():
            return text
        # OCR fallback
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                image_texts = []
                for file in zip_ref.namelist():
                    if file.startswith("word/media/") and file.endswith((".png", ".jpg", ".jpeg")):
                        zip_ref.extract(file, tmpdir)
                        img_path = os.path.join(tmpdir, file)
                        img = Image.open(img_path)
                        image_texts.append(pytesseract.image_to_string(img))
                return "\n".join(image_texts)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    else:
        return ""


# Load feedback CSV
def load_feedback():
    if os.path.exists("feedback_log.csv"):
        return pd.read_csv("feedback_log.csv")
    return pd.DataFrame()


# Match document text to best retention category
def match_retention(text, rules_df, feedback_df=None):
    cleaned_text = text.strip()
    doc_embedding = model.encode(cleaned_text, convert_to_tensor=True)

    # Try to find a similar prior feedback example
    if feedback_df is not None and not feedback_df.empty:
        feedback_texts = feedback_df["document_text"].tolist()
        feedback_embeddings = model.encode(feedback_texts, convert_to_tensor=True)
        similarities = util.cos_sim(doc_embedding, feedback_embeddings)[0]

        # Find the best match
        top_idx = similarities.argmax().item()
        top_score = float(similarities[top_idx])

        if top_score >= 0.80:
            row = feedback_df.iloc[top_idx]
            return {
                "category": row["override_category"],
                "description": f"User-verified category (matched by similarity {top_score:.2f})",
                "retention": row["override_retention"],
                "score": 1.0
            }

    # Fallback to normal embedding similarity with retention categories
    rule_embeddings = model.encode(rules_df["category_description"].tolist(), convert_to_tensor=True)
    similarities = util.cos_sim(doc_embedding, rule_embeddings)[0]
    top_idx = similarities.argmax().item()
    return {
        "category": rules_df.iloc[top_idx]["category_name"],
        "description": rules_df.iloc[top_idx]["category_description"],
        "retention": rules_df.iloc[top_idx]["retention_period"],
        "score": float(similarities[top_idx])
    }


# Confidence level mapping
def get_confidence_level(score):
    if score >= 0.8:
        return "High", "🟢 High (≥ 0.80): Confident match"
    elif score >= 0.6:
        return "Medium", "🟡 Medium (0.60–0.79): Reasonable match"
    else:
        return "Low", "🔴 Low (< 0.60): Needs review"


# Save feedback entries to CSV
def save_feedback_to_csv(entry, csv_path="feedback_log.csv"):
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_new = pd.concat([df_existing, pd.DataFrame([entry])], ignore_index=True)
    else:
        df_new = pd.DataFrame([entry])
    df_new.to_csv(csv_path, index=False)


# Upload retention schedule
retention_file = st.file_uploader("📎 Upload a Retention Schedule (PDF)", type="pdf")
if retention_file:
    retention_df = extract_retention_from_pdf(retention_file)
    st.success(f"✅ Retention schedule '{retention_file.name}' loaded with {len(retention_df)} categories.")

    # Load prior feedback
    feedback_df = load_feedback()

    # Upload documents
    uploaded_files = st.file_uploader(
        "📄 Upload Documents to Classify (PDF, DOCX, TXT, Images)",
        type=["pdf", "docx", "txt", "png", "jpg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 Document: {uploaded_file.name}", expanded=True):
                text = extract_text(uploaded_file)
                if not text.strip():
                    st.warning("⚠️ No readable text found.")
                    continue

                match = match_retention(text, retention_df, feedback_df)
                level, definition = get_confidence_level(match["score"])

                if match["description"].startswith("User-verified"):
                    st.markdown("✅ **Matched from previous feedback**")

                st.markdown(f"**Suggested Category**: {match['category']}")
                st.markdown(f"**Retention Period**: {match['retention']}")
                st.markdown(f"**Description**: {match['description']}")
                st.markdown(f"**Confidence Level**: {definition}")
                st.markdown(f"**Schedule Used**: {retention_file.name}")

                # Inputs to allow edits
                override_category = st.text_input(
                    "Override Category",
                    value=match["category"],
                    key=f"cat_{uploaded_file.name}"
                )
                override_retention = st.text_input(
                    "Override Retention Period",
                    value=match["retention"],
                    key=f"ret_{uploaded_file.name}"
                )

                col1, col2 = st.columns(2)
                if col1.button("✅ Confirm", key=f"confirm_{uploaded_file.name}"):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feedback_entry = {
                        "timestamp": timestamp,
                        "document": uploaded_file.name,
                        "document_text": text.strip(),
                        "original_category": match["category"],
                        "original_retention": match["retention"],
                        "override_category": match["category"],
                        "override_retention": match["retention"],
                        "confidence_level": level,
                        "confidence_score": match["score"],
                        "retention_schedule": retention_file.name
                    }
                    save_feedback_to_csv(feedback_entry)
                    st.success(f"✅ Confirmed and saved for {uploaded_file.name}")

                if col2.button("💾 Save Edits", key=f"save_{uploaded_file.name}"):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    feedback_entry = {
                        "timestamp": timestamp,
                        "document": uploaded_file.name,
                        "document_text": text.strip(),
                        "original_category": match["category"],
                        "original_retention": match["retention"],
                        "override_category": override_category,
                        "override_retention": override_retention,
                        "confidence_level": level,
                        "confidence_score": match["score"],
                        "retention_schedule": retention_file.name
                    }
                    save_feedback_to_csv(feedback_entry)
                    st.success(f"💾 Edited and saved for {uploaded_file.name}")

    # Allow download of feedback log
    if os.path.exists("feedback_log.csv"):
        with open("feedback_log.csv", "rb") as f:
            st.download_button(
                label="⬇️ Download All Feedback as CSV",
                data=f,
                file_name="feedback_log.csv",
                mime="text/csv"
            )
else:
    st.info("👆 Please upload a retention schedule to begin.")
