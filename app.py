import os
import io
import base64
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

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
# Theme / Styling (dark + red accent)
# =========================
def set_theme():
    st.markdown("""
    <style>
    :root {
        --background: #0b0b0c;
        --surface: #161618;
        --input: #242426;
        --border: #2b2b2f;
        --text: #e5e5e7;
        --muted-text: #a0a0a5;
        --accent: #dc2626; /* red-600 */
    }
    .stApp { background: var(--background); color: var(--text); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"]{
        display:flex; gap:6px; padding-bottom:2px;
        border-bottom:1px solid var(--border); align-items:center;
    }
    .stTabs [data-baseweb="tab"]{
        background:#121214; color:var(--muted-text);
        border-radius:6px 6px 0 0; padding:8px 14px;
        border-bottom:2px solid transparent;
    }
    .stTabs [aria-selected="true"]{
        color:var(--text); background:#161618; border-bottom-color:var(--accent);
    }

    /* Inputs */
    .stTextInput > div > div, .stTextArea > div > div,
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--input) !important; border:1px solid var(--border) !important;
        border-radius:8px !important;
    }
    .stTextInput input, .stTextArea textarea {
        color: var(--text) !important; background: var(--input) !important;
    }
    .stMultiSelect [data-baseweb="tag"]{
        background:var(--accent)!important; color:#fff!important; border-radius:6px!important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        transition: border-color .15s ease-in-out;
    }
    .stButton > button:hover { border-color: var(--accent) !important; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# GitHub Helpers (safe if secrets missing)
# =========================
def _gh_available() -> bool:
    try:
        g = st.secrets.get("github", None)
        return bool(g) and {"token", "owner", "repo"}.issubset(g.keys())
    except Exception:
        return False

def _gh_headers() -> Dict[str, str]:
    g = st.secrets["github"]
    return {"Authorization": f"Bearer {g['token']}", "Accept": "application/vnd.github+json"}

def _gh_info() -> Tuple[str, str, str, str]:
    g = st.secrets["github"]
    return g["owner"], g["repo"], g.get("branch", "main"), g.get("path_prefix", "data")

def _gh_api_base(owner: str, repo: str) -> str:
    return f"https://api.github.com/repos/{owner}/{repo}"

def _gh_get_file(owner: str, repo: str, branch: str, path: str):
    url = f"{_gh_api_base(owner, repo)}/contents/{path}?ref={branch}"
    return requests.get(url, headers=_gh_headers())

def _gh_get_file_sha(owner: str, repo: str, branch: str, path: str):
    r = _gh_get_file(owner, repo, branch, path)
    if r.status_code == 200:
        try:
            return r.json().get("sha")
        except Exception:
            return None
    return None

def _gh_read_csv(filename: str):
    if not _gh_available():
        return None
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

def _gh_write_csv(filename: str, df: pd.DataFrame, message: str = "update data") -> bool:
    if not _gh_available():
        raise RuntimeError("GitHub secrets not configured")
    owner, repo, branch, prefix = _gh_info()
    path = f"{prefix}/{filename}"
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    content_b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("utf-8")
    sha = _gh_get_file_sha(owner, repo, branch, path)
    payload = {"message": message, "content": content_b64, "branch": branch}
    if sha:
        payload["sha"] = sha
    url = f"{_gh_api_base(owner, repo)}/contents/{path}"
    r = requests.put(url, headers=_gh_headers(), data=json.dumps(payload))
    r.raise_for_status()
    return True

# =========================
# Data Manager (GitHub-first, local fallback)
# =========================
class DataManager:
    @staticmethod
    def load_data(filename: str, columns: List[str]) -> pd.DataFrame:
        # Try GitHub first
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

        # Fallback local (ephemeral on Streamlit Cloud)
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
        # Try GitHub
        try:
            _gh_write_csv(filename, df, message=f"update {filename}")
            return
        except Exception as e:
            st.warning(f"GitHub save failed for {filename}: {e}")
        # Fallback local
        df.to_csv(filename, index=False)

    @staticmethod
    def save_prompt(name: str, prompt: str) -> None:
        df = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        new_row = pd.DataFrame({'name': [name], 'timestamp': [datetime.now()], 'prompt': [prompt]})
        df = pd.concat([df, new_row], ignore_index=True)
        DataManager.save_data(df, 'prompt_history.csv')

# =========================
# UI Components
# =========================
class ElementCreator:
    @staticmethod
    def render():
        with st.expander("Create New Element", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                element_type = st.selectbox("Type", ELEMENT_TYPES, key="new_type")
                title = st.text_input("Title", key="new_title")
            with col2:
                content = st.text_area("Content", key="new_content", height=100)

            if st.button("Add Element", key="add_element"):
                if not title:
                    st.error("Title is required.")
                else:
                    df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
                    new_row = pd.DataFrame({'title': [title], 'type': [element_type], 'content': [content]})
                    df = pd.concat([df, new_row], ignore_index=True)
                    DataManager.save_data(df, 'prompt_elements.csv')
                    st.success("Element added successfully!")

class ElementEditor:
    @staticmethod
    def render():
        df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
        if df.empty:
            st.warning("No elements found. Please create some elements first.")
            return

        col1, _ = st.columns(2)
        with col1:
            all_types = ['All'] + sorted(df['type'].unique().tolist())
            selected_type = st.selectbox("Filter by Type", all_types, key="filter_type")

        filtered_df = df if selected_type == 'All' else df[df['type'] == selected_type]
        if filtered_df.empty:
            st.warning(f"No elements found for type: {selected_type}")
            return

        for index, row in filtered_df.iterrows():
            with st.expander(f"{row['title']} ({row['type']})", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    nt = st.text_input("Title", value=row['title'], key=f"title_{index}")
                    ntype = st.selectbox("Type", ELEMENT_TYPES, index=ELEMENT_TYPES.index(row['type']), key=f"type_{index}")
                with c2:
                    nc = st.text_area("Content", value=row['content'], key=f"content_{index}", height=100)
                u1, u2 = st.columns(2)
                with u1:
                    if st.button("Update", key=f"update_{index}"):
                        df.at[index, 'title'] = nt
                        df.at[index, 'type'] = ntype
                        df.at[index, 'content'] = nc
                        DataManager.save_data(df, 'prompt_elements.csv')
                        st.success("Updated successfully!")
                with u2:
                    if st.button("Delete", key=f"delete_{index}"):
                        DataManager.save_data(df.drop(index), 'prompt_elements.csv')
                        st.success("Deleted successfully!")

# =========================
# Prompt Builder
# =========================
class PromptBuilder:
    @staticmethod
    def render():
        df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)

        # --- Selection UI (3 columns)
        c1, c2, c3 = st.columns(3)
        selections = {}
        with c1:
            selections['role'] = PromptBuilder._sec("Role", 'role', df)
            selections['goal'] = PromptBuilder._sec("Goal", 'goal', df)
        with c2:
            selections['audience'] = PromptBuilder._sec("Target Audience", 'audience', df, True)
            selections['context']  = PromptBuilder._sec("Context", 'context', df, True)
        with c3:
            selections['output'] = PromptBuilder._sec("Output", 'output', df, True)
            selections['tone']   = PromptBuilder._sec("Tone", 'tone', df)

        # --- Build prompt (checkboxes are at bottom; preserve current values)
        auto = st.session_state.get("auto_update_prompt", True)
        recursive = st.session_state.get("recursive_feedback", False)
        prompt, missing = PromptBuilder._build(selections, df, recursive)

        if missing:
            st.warning(
                "These elements have no **Content** set (using the Title as a fallback). "
                "Add Content in **Element Editor** for better prompts:\n\n- " + "\n- ".join(missing)
            )

        if "generated_prompt" not in st.session_state:
            st.session_state["generated_prompt"] = ""
        if auto:
            st.session_state["generated_prompt"] = prompt

        # --- Name + action buttons row (Name | Save | Clear)
        n_col, save_col, clear_col = st.columns([6, 1, 1], vertical_alignment="bottom")
        with n_col:
            with n_col:
    st.text_input(
        "Prompt Name",
        key="prompt_name",
        placeholder="Enter Prompt Name if you would like to save this prompt."
    )

        with save_col:
            if st.button("Save Prompt", key="save_prompt_btn", use_container_width=True):
                text_to_save = st.session_state.get("generated_prompt", "").strip()
                name = st.session_state.get("prompt_name", "").strip()
                if not text_to_save:
                    st.warning("Nothing to save — the prompt is empty.")
                elif not name:
                    st.warning("Please enter a Prompt Name.")
                else:
                    DataManager.save_prompt(name, text_to_save)
                    st.success(f"Saved: {name}")
        with clear_col:
            st.button("Clear Form", key="clear_form_btn", on_click=clear_form_state, use_container_width=True)

        # --- Prompt editor
        st.text_area("Generated Prompt", height=280, key="generated_prompt")

        # --- Tip + bottom-right toggles
        tip_col, toggles_col = st.columns([6, 2])
        with tip_col:
            st.info("Tip: turn OFF Auto-update if you want to manually edit and keep your edits while changing selections.")
        with toggles_col:
            t1, t2 = st.columns(2)
            with t1:
                st.checkbox("Auto-update", key="auto_update_prompt", value=auto)
            with t2:
                st.checkbox("Request recursive feedback", key="recursive_feedback", value=recursive)

    @staticmethod
    def _sec(title: str, element_type: str, df: pd.DataFrame, multi: bool = False) -> Dict[str, Any]:
        """Render a selectbox/multiselect for a section, with 'Skip' and 'Write your own'."""
        elements = df[df['type'] == element_type]
        options = ["Skip", "Write your own"] + elements['title'].tolist()

        if multi:
            selected = st.multiselect(title, options, key=f"select_{element_type}")
        else:
            selected = st.selectbox(title, options, key=f"select_{element_type}")

        custom_content = ""
        if (multi and isinstance(selected, list) and "Write your own" in selected) or \
           (not multi and selected == "Write your own"):
            custom_content = st.text_input(f"Custom {title}", key=f"custom_{element_type}")

        return {"selected": selected, "custom": custom_content, "elements": elements}

    @staticmethod
    def _content_or_title(df: pd.DataFrame, title: str) -> Tuple[str, str]:
        """Return (content, missing_label). If content empty, use title and mark missing."""
        row = df[df['title'] == title]
        if row.empty:
            return "", ""
        c = str(row['content'].values[0]).strip()
        return (c, "") if c else (title, title)

    @staticmethod
    def _build(selections: Dict[str, Dict], df: pd.DataFrame, recursive: bool) -> Tuple[str, list]:
        parts, missing = [], []

        for section, data in selections.items():
            sel = data['selected']
            if (isinstance(sel, str) and sel == "Skip") or (isinstance(sel, list) and (not sel or sel == ["Skip"])):
                continue

            title = "Target Audience" if section == "audience" else section.title()

            # Multi sections
            if section in ['audience', 'context', 'output']:
                if data['custom']:
                    content = data['custom']
                elif isinstance(sel, list):
                    snips = []
                    for t in [s for s in sel if s not in ("Skip", "Write your own")]:
                        c, m = PromptBuilder._content_or_title(df, t)
                        if m: missing.append(f"{title} → {m}")
                        if c: snips.append(c)
                    content = "\n".join(snips)
                else:
                    if sel not in ("Skip", "Write your own"):
                        c, m = PromptBuilder._content_or_title(df, sel)
                        if m: missing.append(f"{title} → {m}")
                        content = c
                    else:
                        content = ""
                if content:
                    parts.append(f"{title}:\n{content}")

            # Single sections
            else:
                if isinstance(sel, str) and sel == "Write your own":
                    content = data['custom']
                elif isinstance(sel, str):
                    c, m = PromptBuilder._content_or_title(df, sel)
                    if m: missing.append(f"{title} → {m}")
                    content = c
                else:
                    content = ""
                if content:
                    parts.append(f"{title}: {content}")

        prompt = "\n\n".join(parts)
        if recursive and prompt:
            prompt += (
                "\n\nBefore you provide the response, please ask me any questions that you feel could "
                "help you craft a better response. If you feel you have enough information to craft this response, "
                "please just provide it."
            )
        return prompt, missing

# =========================
# Clear form helper (resets selects/multiselects/customs/prompt name + text)
# =========================
def clear_form_state():
    defaults = {
        "select_role": "Skip",
        "select_goal": "Skip",
        "select_tone": "Skip",
        "select_audience": [],
        "select_context": [],
        "select_output": [],
        "custom_role": "",
        "custom_goal": "",
        "custom_tone": "",
        "custom_audience": "",
        "custom_context": "",
        "custom_output": "",
        "generated_prompt": "",
        "prompt_name": "",
    }
    for k, v in defaults.items():
        st.session_state[k] = v
    # Leave 'auto_update_prompt' and 'recursive_feedback' untouched.

# =========================
# Prompt Browser
# =========================
class PromptBrowser:
    @staticmethod
    def render():
        df = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        if df.empty:
            st.warning("No prompts found. Please create and save some prompts first.")
            return
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
        except Exception:
            pass
        for i, row in df.iterrows():
            with st.expander(f"{row['name']} - {row['timestamp']}", expanded=False):
                st.text_area("Prompt Content", value=row['prompt'], height=150, key=f"prompt_{i}")

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

    tabs = st.tabs([
        "Element Creator", "Element Editor", "Prompt Builder",
        "Browse Prompts", "Backup / Restore"
    ])
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


