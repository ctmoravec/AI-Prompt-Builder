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
# Theme / Styling (dark with red accent)
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
        gap:6px; padding-bottom:2px; border-bottom:1px solid var(--border);
        align-items: center;
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
    .stTextInput input, .stTextArea textarea { color: var(--text) !important; background: var(--input) !important; }
    .stMultiSelect [data-baseweb="tag"]{ background:var(--accent)!important; color:#fff!important; border-radius:6px!important; }

    /* Faux-tab button on header row (right side) */
    .tabbar-btn > button{
        background:#121214 !important; color:var(--muted-text) !important;
        border:0 !important; border-bottom:2px solid transparent !important;
        border-radius:6px 6px 0 0 !important; padding:8px 14px !important;
    }
    .tabbar-btn > button:hover{ color:var(--text) !important; border-bottom-color:var(--accent) !important; }
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
                        df = df.drop(index)
                        DataManager.save_data(df, 'prompt_elements.csv')
                        st.success("Deleted successfully!")

class PromptBuilder:
    @staticmethod
    def render():
        df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)

        col1, col2, col3 = st.columns(3)
        selections = {}
        with col1:
            selections['role'] = PromptBuilder._sec("Role", 'role', df)
            selections['goal'] = PromptBuilder._sec("Goal", 'goal', df)
        with col2:
            selections['audience'] = PromptBuilder._sec("Target Audience", 'audience', df, True)
            selections['context'] = PromptBuilder._sec("Context", 'context', df, True)
        with col3:
            selections['output'] = PromptBuilder._sec("Output", 'output', df, True)
            selections['tone'] = PromptBuilder._sec("Tone", 'tone', df)

        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            auto = st.checkbox("Auto-update", value=True, key="auto_update_prompt")
        with c2:
            recursive = st.checkbox("Request recursive feedback", value=False, key="recursive_feedback")

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

        st.text_area("Generated Prompt", height=250, key="generated_prompt")
        st.info("Tip: turn OFF Auto-update if you want to manually edit and keep your edits while changing selections.")

        s1, s2 = st.columns([2, 1])
        with s1:
            pname = st.text_input("Prompt Name", key="prompt_name")
        with s2:
            if st.button("Save Prompt", key="save_prompt_btn"):
                txt = st.session_state.get("generated_prompt", "").strip()
                if not txt:
                    st.warning("Nothing to save — the prompt is empty.")
                elif not pname:
                    st.warning("Please enter a Prompt Name.")
                else:
                    DataManager.save_prompt(pname, txt)
                    st.success(f"Saved: {pname}")

    @staticmethod
    def _sec(title: str, etype: str, df: pd.DataFrame, multi: bool = False) -> Dict[str, Any]:
        elements = df[df['type'] == etype]
        options = ["Skip", "Write your own"] + elements['title'].tolist()

        if multi:
            selected = st.multiselect(title, options, key=f"select_{etype}")
        else:
            selected = st.selectbox(title, options, key=f"select_{etype}")

        custom = ""
        if (multi and isinstance(selected, list) and "Write your own" in selected) or \
           (not multi and selected == "Write your own"):
            custom = st.text_input(f"Custom {title}", key=f"custom_{etype}")

        return {'selected': selected, 'custom': custom, 'elements': elements}

    @staticmethod
    def _content_or_title(df: pd.DataFrame, title: str) -> Tuple[str, str]:
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

            title = "Target Audience" if section == 'audience' else section.title()

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
# Clear Form (button-only; top row; clears everything including checkboxes)
# =========================
CLEAR_KEYS = [
    # dropdowns / multiselects
    "select_role","select_goal","select_audience","select_context","select_output","select_tone",
    # custom inputs
    "custom_role","custom_goal","custom_audience","custom_context","custom_output","custom_tone",
    # checkboxes
    "auto_update_prompt","recursive_feedback",
    # prompt area + name
    "generated_prompt","prompt_name",
]

def clear_form_state():
    for k in CLEAR_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    # no explicit st.rerun(); Streamlit reruns automatically after a button click

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
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Download current data**")
        elements = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
        b1 = io.StringIO(); elements.to_csv(b1, index=False)
        st.download_button("Download prompt_elements.csv", data=b1.getvalue(), file_name="prompt_elements.csv", mime="text/csv", key="dl_elements")

        history = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        b2 = io.StringIO(); history.to_csv(b2, index=False)
        st.download_button("Download prompt_history.csv", data=b2.getvalue(), file_name="prompt_history.csv", mime="text/csv", key="dl_history")

    with c2:
        st.markdown("**Restore / merge from a backup**")
        up1 = st.file_uploader("Upload prompt_elements.csv", type=["csv"], key="up_elements")
        if up1 is not None:
            try:
                new_df = pd.read_csv(up1)
                base_df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
                comb = pd.concat([base_df, new_df], ignore_index=True).drop_duplicates(subset=CSV_COLUMNS, keep="last")
                DataManager.save_data(comb, 'prompt_elements.csv')
                st.success("Elements merged and saved.")
            except Exception as e:
                st.error(f"Failed to import elements: {e}")

        up2 = st.file_uploader("Upload prompt_history.csv", type=["csv"], key="up_history")
        if up2 is not None:
            try:
                new_h = pd.read_csv(up2)
                base_h = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
                comb = pd.concat([base_h, new_h], ignore_index=True).drop_duplicates(subset=PROMPT_HISTORY_COLUMNS, keep="last")
                DataManager.save_data(comb, 'prompt_history.csv')
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

    # Header row: tabs on the left, Clear button on the right (same row)
    header_l, header_r = st.columns([8, 1], vertical_alignment="bottom")
    with header_l:
        tabs = st.tabs(["Element Creator","Element Editor","Prompt Builder","Browse Prompts","Backup / Restore"])
    with header_r:
        st.button("Clear Form", key="clear_form_btn", on_click=clear_form_state, help="Clear all inputs", type="secondary", use_container_width=True)
        st.markdown('<div class="tabbar-btn"></div>', unsafe_allow_html=True)  # style hook

    # Tabs content
    with tabs[0]: ElementCreator.render()
    with tabs[1]: ElementEditor.render()
    with tabs[2]: PromptBuilder.render()
    with tabs[3]: PromptBrowser.render()
    with tabs[4]: render_backup_restore_tab()

if __name__ == "__main__":
    main()
