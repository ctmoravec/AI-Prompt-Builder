import os
import io
import base64
import json
from datetime import datetime
from typing import List, Dict, Any

import requests
import streamlit as st
import pandas as pd

# =========================
# Constants
# =========================
ELEMENT_TYPES = ['role', 'goal', 'audience', 'context', 'output', 'tone']
CSV_COLUMNS = ['title', 'type', 'content']
PROMPT_HISTORY_COLUMNS = ['name', 'timestamp', 'prompt']

# =========================
# Theme / Styling
# =========================
def set_theme():
    st.markdown("""
    <style>
    :root {
        --background: #09090B;
        --foreground: #FAFAFA;
        --muted: #27272A;
        --muted-foreground: #A1A1AA;
        --popover: #18181B;
        --border: #27272A;
        --input: #27272A;
        --primary: #FAFAFA;
        --secondary: #27272A;
    }
    .stApp { background-color: var(--background); color: var(--foreground); }
    .stTitle { color: var(--foreground) !important; font-weight: 600 !important; }
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stSelectbox > div > div {
        background-color: var(--input) !important;
        border-color: var(--border) !important;
        border-radius: 6px !important;
    }
    .stButton > button {
        background-color: var(--secondary) !important;
        color: var(--foreground) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: var(--muted) !important;
        border-color: var(--primary) !important;
    }
    .streamlit-expanderHeader {
        background-color: var(--secondary) !important;
        border-color: var(--border) !important;
        border-radius: 6px !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1px; background-color: var(--background); }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--secondary);
        border-radius: 4px 4px 0 0;
        padding: 8px 16px;
        color: var(--muted-foreground);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--muted);
        color: var(--foreground);
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# GitHub Helpers
# =========================
def _gh_headers():
    return {
        "Authorization": f"Bearer {st.secrets['github']['token']}",
        "Accept": "application/vnd.github+json",
    }

def _gh_info():
    g = st.secrets["github"]
    return g["owner"], g["repo"], g.get("branch", "main"), g.get("path_prefix", "data")

def _gh_api_base(owner, repo):
    return f"https://api.github.com/repos/{owner}/{repo}"

def _gh_get_file(owner, repo, branch, path):
    url = f"{_gh_api_base(owner, repo)}/contents/{path}?ref={branch}"
    r = requests.get(url, headers=_gh_headers())
    return r

def _gh_get_file_sha(owner, repo, branch, path):
    r = _gh_get_file(owner, repo, branch, path)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def _gh_read_csv(filename):
    try:
        owner, repo, branch, prefix = _gh_info()
        path = f"{prefix}/{filename}"
        r = _gh_get_file(owner, repo, branch, path)
        if r.status_code == 200:
            content_b64 = r.json()["content"]
            content = base64.b64decode(content_b64).decode("utf-8")
            return io.StringIO(content)
        return None
    except Exception:
        return None

def _gh_write_csv(filename, df: pd.DataFrame, message="update data"):
    owner, repo, branch, prefix = _gh_info()
    path = f"{prefix}/{filename}"
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    content_b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("utf-8")

    sha = _gh_get_file_sha(owner, repo, branch, path)
    payload = {
        "message": message,
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    url = f"{_gh_api_base(owner, repo)}/contents/{path}"
    r = requests.put(url, headers=_gh_headers(), data=json.dumps(payload))
    r.raise_for_status()
    return True

# =========================
# Data Manager (GitHub + local fallback)
# =========================
class DataManager:
    @staticmethod
    def load_data(filename: str, columns: List[str]) -> pd.DataFrame:
        try:
            buf = _gh_read_csv(filename)
            if buf is not None:
                df = pd.read_csv(buf)
                for c in columns:
                    if c not in df.columns:
                        df[c] = pd.Series(dtype="object")
                return df[columns] if not df.empty else pd.DataFrame(columns=columns)
        except Exception as e:
            st.warning(f"GitHub load failed for {filename}: {e}")

        # Fallback local
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=columns)
            df.to_csv(filename, index=False)
            return df
        df = pd.read_csv(filename)
        for c in columns:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
        return df[columns] if not df.empty else pd.DataFrame(columns=columns)

    @staticmethod
    def save_data(df: pd.DataFrame, filename: str) -> None:
        try:
            _gh_write_csv(filename, df, message=f"update {filename}")
            return
        except Exception as e:
            st.warning(f"GitHub save failed for {filename}: {e}")
        df.to_csv(filename, index=False)

    @staticmethod
    def save_prompt(name: str, prompt: str) -> None:
        df = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        new_row = pd.DataFrame({'name': [name], 'timestamp': [datetime.now()], 'prompt': [prompt]})
        df = pd.concat([df, new_row], ignore_index=True)
        DataManager.save_data(df, 'prompt_history.csv')

# =========================
# (UI classes: ElementCreator, ElementEditor, PromptBuilder, PromptBrowser)
# =========================
# 🔹 Keep all your existing ElementCreator, ElementEditor, PromptBuilder, PromptBrowser classes unchanged.
# (Paste them here exactly as you had them)

# =========================
# Backup / Restore Tab
# =========================
def render_backup_restore_tab():
    st.subheader("Backup / Restore")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Download current data**")
        elements_df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
        buf1 = io.StringIO()
        elements_df.to_csv(buf1, index=False)
        st.download_button("Download prompt_elements.csv", data=buf1.getvalue(),
                           file_name="prompt_elements.csv", mime="text/csv", key="dl_elements")

        history_df = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        buf2 = io.StringIO()
        history_df.to_csv(buf2, index=False)
        st.download_button("Download prompt_history.csv", data=buf2.getvalue(),
                           file_name="prompt_history.csv", mime="text/csv", key="dl_history")

    with col2:
        st.markdown("**Restore / merge from a backup**")
        up1 = st.file_uploader("Upload prompt_elements.csv", type=["csv"], key="up_elements")
        if up1 is not None:
            try:
                new_df = pd.read_csv(up1)
                base_df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
                combined = pd.concat([base_df, new_df], ignore_index=True)
                combined.drop_duplicates(subset=CSV_COLUMNS, keep="last", inplace=True)
                DataManager.save_data(combined, 'prompt_elements.csv')
                st.success("Elements merged and saved.")
            except Exception as e:
                st.error(f"Failed to import elements: {e}")

        up2 = st.file_uploader("Upload prompt_history.csv", type=["csv"], key="up_history")
        if up2 is not None:
            try:
                new_hist = pd.read_csv(up2)
                base_hist = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
                combined = pd.concat([base_hist, new_hist], ignore_index=True)
                combined.drop_duplicates(subset=PROMPT_HISTORY_COLUMNS, keep="last", inplace=True)
                DataManager.save_data(combined, 'prompt_history.csv')
                st.success("History merged and saved.")
            except Exception as e:
                st.error(f"Failed to import history: {e}")

# =========================
# Main
# =========================
def main():
    st.set_page_config(layout="wide", page_title="CTM Enterprises Prompt Creation Tool")
    set_theme()
    st.title("CTM Enterprises Prompt Creation Tool")

    tabs = st.tabs(["Element Creator", "Element Editor", "Prompt Builder", "Browse Prompts", "Backup / Restore"])
    with tabs[0]:
        ElementCreator.render()
    with tabs[1]:
        ElementEditor.render()
    with tabs[2]:
        PromptBuilder.render()
    with tabs[3]:
        PromptBrowser.render()
    with tabs[4]:
        render_backup_restore_tab()

if __name__ == "__main__":
    main()
