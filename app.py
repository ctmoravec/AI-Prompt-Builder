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
    .stTextInput input, .stTextArea textarea {
        color: var(--foreground) !important;
        background-color: var(--input) !important;
    }

    .stButton > button {
        background-color: var(--secondary) !important;
        color: var(--foreground) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        transition: all 0.2s ease-in-out !important;
        padding: 8px 16px !important;
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
# GitHub Helpers (safe if secrets missing)
# =========================
def _gh_available() -> bool:
    try:
        g = st.secrets.get("github", None)
        if not g:
            return False
        needed = {"token", "owner", "repo"}
        return needed.issubset(set(g.keys()))
    except Exception:
        return False

def _gh_headers() -> Dict[str, str]:
    g = st.secrets["github"]
    return {
        "Authorization": f"Bearer {g['token']}",
        "Accept": "application/vnd.github+json",
    }

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
                    return
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
            ElementEditor._render_element(index, row, df)

    @staticmethod
    def _render_element(index: int, row: Dict[str, Any], df: pd.DataFrame):
        with st.expander(f"{row['title']} ({row['type']})", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_title = st.text_input("Title", value=row['title'], key=f"title_{index}")
                new_type = st.selectbox("Type", ELEMENT_TYPES,
                                        index=ELEMENT_TYPES.index(row['type']),
                                        key=f"type_{index}")
            with col2:
                new_content = st.text_area("Content", value=row['content'],
                                           key=f"content_{index}", height=100)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update", key=f"update_{index}"):
                    df.at[index, 'title'] = new_title
                    df.at[index, 'type'] = new_type
                    df.at[index, 'content'] = new_content
                    DataManager.save_data(df, 'prompt_elements.csv')
                    st.success("Updated successfully!")
                    st.rerun()
            with col2:
                if st.button("Delete", key=f"delete_{index}"):
                    df = df.drop(index)
                    DataManager.save_data(df, 'prompt_elements.csv')
                    st.success("Deleted successfully!")
                    st.rerun()

class PromptBuilder:
    @staticmethod
    def render():
        df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)

        # --- Selection UI
        col1, col2, col3 = st.columns(3)
        selections = {}
        with col1:
            selections['role'] = PromptBuilder._create_section("Role", 'role', df)
            selections['goal'] = PromptBuilder._create_section("Goal", 'goal', df)
        with col2:
            selections['audience'] = PromptBuilder._create_section("Target Audience", 'audience', df, True)
            selections['context'] = PromptBuilder._create_section("Context", 'context', df, True)
        with col3:
            selections['output'] = PromptBuilder._create_section("Output", 'output', df, True)
            selections['tone'] = PromptBuilder._create_section("Tone", 'tone', df)

        # --- Controls
        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            auto = st.checkbox("Auto-update", value=True, key="auto_update_prompt")
        with c2:
            recursive_feedback = st.checkbox("Request recursive feedback", value=False, key="recursive_feedback")

        # Compute prompt + missing warnings
        prompt, missing = PromptBuilder._generate_prompt(selections, df, recursive_feedback)

        if missing:
            st.warning(
                "These elements have no **Content** set (using the Title as a fallback). "
                "Add Content in **Element Editor** for better prompts:\n\n- " + "\n- ".join(missing)
            )

        # --- Keep session_state in sync if Auto-update is ON
        if "generated_prompt" not in st.session_state:
            st.session_state["generated_prompt"] = ""
        if auto:
            st.session_state["generated_prompt"] = prompt

        # --- Prompt editor (stateful so the user can tweak/copy)
        st.text_area("Generated Prompt", height=250, key="generated_prompt")
        st.info("Tip: turn OFF Auto-update if you want to manually edit and keep your edits while changing selections.")

        # --- Save controls (save whatever is currently in the box)
        csave1, csave2 = st.columns([2, 1])
        with csave1:
            prompt_name = st.text_input("Prompt Name", key="prompt_name")
        with csave2:
            if st.button("Save Prompt", key="save_prompt_btn"):
                text_to_save = st.session_state.get("generated_prompt", "").strip()
                if not text_to_save:
                    st.warning("Nothing to save — the prompt is empty.")
                elif not prompt_name:
                    st.warning("Please enter a Prompt Name.")
                else:
                    DataManager.save_prompt(prompt_name, text_to_save)
                    st.success(f"Saved: {prompt_name}")

    @staticmethod
    def _create_section(title: str, element_type: str, df: pd.DataFrame,
                        multi_select: bool = False) -> Dict[str, Any]:
        elements = df[df['type'] == element_type]
        options = ["Skip", "Write your own"] + elements['title'].tolist()

        if multi_select:
            selected = st.multiselect(title, options, key=f"select_{element_type}")
        else:
            selected = st.selectbox(title, options, key=f"select_{element_type}")

        custom_content = ""
        if (multi_select and isinstance(selected, list) and "Write your own" in selected) or \
           (not multi_select and selected == "Write your own"):
            custom_content = st.text_input(f"Custom {title}", key=f"custom_{element_type}")

        return {'selected': selected, 'custom': custom_content, 'elements': elements}

    @staticmethod
    def _row_content_or_fallback(df: pd.DataFrame, title: str) -> Tuple[str, str]:
        """Return (content, missing_label). If content is empty, use title as fallback and return the title in missing_label."""
        row = df[df['title'] == title]
        if row.empty:
            return "", ""
        content = str(row['content'].values[0]).strip()
        if content:
            return content, ""
        return title, title  # fallback to title, and report missing

    @staticmethod
    def _generate_prompt(selections: Dict[str, Dict], df: pd.DataFrame,
                         recursive_feedback: bool) -> Tuple[str, list]:
        parts = []
        missing = []

        for section, data in selections.items():
            sel = data['selected']

            # Skip no-choice
            if (isinstance(sel, str) and sel == "Skip") or (isinstance(sel, list) and (not sel or sel == ["Skip"])):
                continue

            title = section.title()
            if section == 'audience':
                title = "Target Audience"

            # Multi-select sections
            if section in ['audience', 'context', 'output']:
                if data['custom']:
                    content = data['custom']
                elif isinstance(sel, list):
                    snippets = []
                    for t in [s for s in sel if s not in ("Skip", "Write your own")]:
                        c, m = PromptBuilder._row_content_or_fallback(df, t)
                        if m:
                            missing.append(f"{title} → {m}")
                        if c:
                            snippets.append(c)
                    content = "\n".join(snippets)
                else:
                    if sel not in ("Skip", "Write your own"):
                        c, m = PromptBuilder._row_content_or_fallback(df, sel)
                        if m:
                            missing.append(f"{title} → {m}")
                        content = c
                    else:
                        content = ""
                if content:
                    parts.append(f"{title}:\n{content}")

            # Single-select sections
            else:
                if isinstance(sel, str) and sel == "Write your own":
                    content = data['custom']
                elif isinstance(sel, str):
                    c, m = PromptBuilder._row_content_or_fallback(df, sel)
                    if m:
                        missing.append(f"{title} → {m}")
                    content = c
                else:
                    content = ""
                if content:
                    parts.append(f"{title}: {content}")

        prompt = "\n\n".join(parts)
        if recursive_feedback:
            prompt += (
                "\n\nBefore you provide the response, please ask me any questions that you feel could "
                "help you craft a better response. If you feel you have enough information to craft this response, "
                "please just provide it."
            )
        return prompt, missing

# =========================
# Clear form helper
# =========================
def clear_form_state():
    keys_to_clear = [
        # selections
        "select_role", "select_goal", "select_audience",
        "select_context", "select_output", "select_tone",
        # custom inputs
        "custom_role", "custom_goal", "custom_audience",
        "custom_context", "custom_output", "custom_tone",
        # controls + output
        "auto_update_prompt", "recursive_feedback", "generated_prompt", "prompt_name"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

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

        for index, row in df.iterrows():
            with st.expander(f"{row['name']} - {row['timestamp']}", expanded=False):
                st.text_area("Prompt Content", value=row['prompt'], height=150, key=f"prompt_{index}")

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

    # Top utility row: single Clear Form button
    left, right = st.columns([8, 1])
    with right:
        st.button("Clear Form", key="clear_form_top", on_click=clear_form_state)

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
