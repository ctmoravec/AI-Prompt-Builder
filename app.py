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
# Theme / Styling (dark + red accent; clear button styled as a tab)
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

    /* Clear button that LOOKS like a tab, sits right after the real tabs */
    .tab-clear-wrap{
        display:inline-block; margin-left:6px; margin-top:-42px; /* tweak -40..-44 to align */
    }
    .tab-clear-wrap > button{
        background:#121214 !important; color:var(--text) !important;
        border:0 !important; border-bottom:2px solid var(--accent) !important;
        border-radius:6px 6px 0 0 !important; padding:8px 14px !important;
    }
    .tab-clear-wrap > button:focus{ outline:none !important; }

    /* Action buttons row: tighter buttons */
    .stButton > button {
        padding: 8px 14px !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)



# =========================
# GitHub Helpers (safe if secrets missing)
# =========================
def _gh_available() -> bool:
    try:
        g = st.secrets.get("github", None)
        return bool(g) and {"token","owner","repo"}.issubset(g.keys())
    except Exception:
        return False

def _gh_headers() -> Dict[str,str]:
    g = st.secrets["github"]
    return {"Authorization": f"Bearer {g['token']}", "Accept":"application/vnd.github+json"}

def _gh_info() -> Tuple[str,str,str,str]:
    g = st.secrets["github"]
    return g["owner"], g["repo"], g.get("branch","main"), g.get("path_prefix","data")

def _gh_api_base(owner:str, repo:str)->str: return f"https://api.github.com/repos/{owner}/{repo}"
def _gh_get_file(owner,repo,branch,path): return requests.get(f"{_gh_api_base(owner,repo)}/contents/{path}?ref={branch}", headers=_gh_headers())
def _gh_get_file_sha(owner,repo,branch,path):
    r=_gh_get_file(owner,repo,branch,path)
    return r.json().get("sha") if r.status_code==200 else None

def _gh_read_csv(filename:str):
    if not _gh_available(): return None
    try:
        owner,repo,branch,prefix=_gh_info()
        r=_gh_get_file(owner,repo,branch,f"{prefix}/{filename}")
        if r.status_code==200:
            from base64 import b64decode
            return io.StringIO(b64decode(r.json()["content"]).decode("utf-8"))
    except Exception: pass
    return None

def _gh_write_csv(filename:str, df:pd.DataFrame, message:str="update data")->bool:
    if not _gh_available(): raise RuntimeError("GitHub secrets not configured")
    owner,repo,branch,prefix=_gh_info()
    path=f"{prefix}/{filename}"
    buf=io.StringIO(); df.to_csv(buf,index=False)
    content=base64.b64encode(buf.getvalue().encode("utf-8")).decode("utf-8")
    payload={"message":message,"content":content,"branch":branch}
    sha=_gh_get_file_sha(owner,repo,branch,path)
    if sha: payload["sha"]=sha
    r=requests.put(f"{_gh_api_base(owner,repo)}/contents/{path}", headers=_gh_headers(), data=json.dumps(payload))
    r.raise_for_status(); return True

# =========================
# Data Manager (GitHub-first, local fallback)
# =========================
class DataManager:
    @staticmethod
    def load_data(filename:str, columns:List[str])->pd.DataFrame:
        try:
            buf=_gh_read_csv(filename)
            if buf is not None:
                df=pd.read_csv(buf)
                for c in columns:
                    if c not in df.columns: df[c]=pd.Series(dtype="object")
                return df[columns] if not df.empty else pd.DataFrame(columns=columns)
        except Exception as e:
            st.warning(f"GitHub load failed for {filename}: {e}")
        if not os.path.exists(filename):
            df=pd.DataFrame(columns=columns); df.to_csv(filename,index=False); return df
        df=pd.read_csv(filename)
        for c in columns:
            if c not in df.columns: df[c]=pd.Series(dtype="object")
        return df[columns] if not df.empty else pd.DataFrame(columns=columns)

    @staticmethod
    def save_data(df:pd.DataFrame, filename:str)->None:
        try:
            _gh_write_csv(filename, df, message=f"update {filename}"); return
        except Exception as e:
            st.warning(f"GitHub save failed for {filename}: {e}")
        df.to_csv(filename, index=False)

    @staticmethod
    def save_prompt(name:str, prompt:str)->None:
        df=DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        new=pd.DataFrame({'name':[name],'timestamp':[datetime.now()],'prompt':[prompt]})
        DataManager.save_data(pd.concat([df,new],ignore_index=True),'prompt_history.csv')

# =========================
# UI Components
# =========================
class ElementCreator:
    @staticmethod
    def render():
        with st.expander("Create New Element", expanded=False):
            c1,c2=st.columns(2)
            with c1:
                etype=st.selectbox("Type", ELEMENT_TYPES, key="new_type")
                title=st.text_input("Title", key="new_title")
            with c2:
                content=st.text_area("Content", key="new_content", height=100)
            if st.button("Add Element", key="add_element"):
                if not title: st.error("Title is required.")
                else:
                    df=DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
                    newrow=pd.DataFrame({'title':[title],'type':[etype],'content':[content]})
                    DataManager.save_data(pd.concat([df,newrow],ignore_index=True),'prompt_elements.csv')
                    st.success("Element added successfully!")

class ElementEditor:
    @staticmethod
    def render():
        df=DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
        if df.empty: st.warning("No elements found. Please create some elements first."); return
        c1,_=st.columns(2)
        with c1:
            all_types=['All']+sorted(df['type'].unique().tolist())
            selected=st.selectbox("Filter by Type", all_types, key="filter_type")
        view=df if selected=='All' else df[df['type']==selected]
        if view.empty: st.warning(f"No elements found for type: {selected}"); return
        for idx,row in view.iterrows():
            with st.expander(f"{row['title']} ({row['type']})", expanded=False):
                c1,c2=st.columns(2)
                with c1:
                    nt=st.text_input("Title", value=row['title'], key=f"title_{idx}")
                    ntype=st.selectbox("Type", ELEMENT_TYPES, index=ELEMENT_TYPES.index(row['type']), key=f"type_{idx}")
                with c2:
                    nc=st.text_area("Content", value=row['content'], key=f"content_{idx}", height=100)
                u1,u2=st.columns(2)
                with u1:
                    if st.button("Update", key=f"update_{idx}"):
                        df.at[idx,'title']=nt; df.at[idx,'type']=ntype; df.at[idx,'content']=nc
                        DataManager.save_data(df,'prompt_elements.csv'); st.success("Updated successfully!")
                with u2:
                    if st.button("Delete", key=f"delete_{idx}"):
                        DataManager.save_data(df.drop(idx),'prompt_elements.csv'); st.success("Deleted successfully!")

class PromptBuilder:
    @staticmethod
    def render():
        df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)

        # --- Selection UI (3 cols, like your sketch)
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

        # --- Build prompt (checks live at the bottom now)
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
        n_col, save_col, clear_col = st.columns([6, 1, 1])
        with n_col:
            st.text_input("Prompt Name", key="prompt_name")
        with save_col:
            if st.button("Save Prompt", key="save_prompt_btn", use_container_width=True):
                text_to_save = st.session_state.get("generated_prompt", "").strip()
                if not text_to_save:
                    st.warning("Nothing to save — the prompt is empty.")
                elif not st.session_state.get("prompt_name", "").strip():
                    st.warning("Please enter a Prompt Name.")
                else:
                    DataManager.save_prompt(st.session_state["prompt_name"].strip(), text_to_save)
                    st.success(f"Saved: {st.session_state['prompt_name'].strip()}")
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

    # (keep your _sec, _content_or_title, _build helpers as-is)


# =========================
# Clear Form (no navigation; resets selects/multiselects/customs/prompt)
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
    for k,v in defaults.items():
        st.session_state[k] = v
    # leave 'auto_update_prompt' and 'recursive_feedback' as-is

# =========================
# Prompt Browser
# =========================
class PromptBrowser:
    @staticmethod
    def render():
        df=DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        if df.empty: st.warning("No prompts found. Please create and save some prompts first."); return
        try: df["timestamp"]=pd.to_datetime(df["timestamp"]); df=df.sort_values("timestamp", ascending=False)
        except Exception: pass
        for i,row in df.iterrows():
            with st.expander(f"{row['name']} - {row['timestamp']}", expanded=False):
                st.text_area("Prompt Content", value=row['prompt'], height=150, key=f"prompt_{i}")

# =========================
# Backup / Restore
# =========================
def render_backup_restore_tab():
    st.subheader("Backup / Restore")
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Download current data**")
        elements=DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
        b1=io.StringIO(); elements.to_csv(b1,index=False)
        st.download_button("Download prompt_elements.csv", data=b1.getvalue(), file_name="prompt_elements.csv", mime="text/csv", key="dl_elements")
        history=DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        b2=io.StringIO(); history.to_csv(b2,index=False)
        st.download_button("Download prompt_history.csv", data=b2.getvalue(), file_name="prompt_history.csv", mime="text/csv", key="dl_history")
    with c2:
        st.markdown("**Restore / merge from a backup**")
        up1=st.file_uploader("Upload prompt_elements.csv", type=["csv"], key="up_elements")
        if up1 is not None:
            try:
                new=pd.read_csv(up1); base=DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
                comb=pd.concat([base,new],ignore_index=True).drop_duplicates(subset=CSV_COLUMNS, keep="last")
                DataManager.save_data(comb,'prompt_elements.csv'); st.success("Elements merged and saved.")
            except Exception as e: st.error(f"Failed to import elements: {e}")
        up2=st.file_uploader("Upload prompt_history.csv", type=["csv"], key="up_history")
        if up2 is not None:
            try:
                newh=pd.read_csv(up2); baseh=DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
                comb=pd.concat([baseh,newh],ignore_index=True).drop_duplicates(subset=PROMPT_HISTORY_COLUMNS, keep="last")
                DataManager.save_data(comb,'prompt_history.csv'); st.success("History merged and saved.")
            except Exception as e: st.error(f"Failed to import history: {e}")

# =========================
# Main
# =========================
def main():
    st.set_page_config(layout="wide", page_title="CTM Enterprises Prompt Creation Tool")
    set_theme()
    st.title("CTM Enterprises Prompt Creation Tool")

    # Real tabs
    tabs = st.tabs(["Element Creator","Element Editor","Prompt Builder","Browse Prompts","Backup / Restore"])

    # Immediately render a button that looks/aligns like a tab (no navigation)
    st.markdown('<div class="tab-clear-wrap">', unsafe_allow_html=True)
    st.button("Clear Form", key="clear_form_btn", on_click=clear_form_state, type="secondary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Tab contents
    with tabs[0]: ElementCreator.render()
    with tabs[1]: ElementEditor.render()
    with tabs[2]: PromptBuilder.render()
    with tabs[3]: PromptBrowser.render()
    with tabs[4]: render_backup_restore_tab()

if __name__ == "__main__":
    main()

