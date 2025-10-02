import os
import io
from datetime import datetime
from typing import List, Dict, Any

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
# Data Manager (local CSV)
# =========================
class DataManager:
    @staticmethod
    def load_data(filename: str, columns: List[str]) -> pd.DataFrame:
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=columns)
            df.to_csv(filename, index=False)
            return df
        df = pd.read_csv(filename)
        # Ensure schema if someone uploaded a file with missing columns
        for c in columns:
            if c not in df.columns:
                df[c] = pd.Series(dtype="object")
        return df[columns] if not df.empty else pd.DataFrame(columns=columns)

    @staticmethod
    def save_data(df: pd.DataFrame, filename: str) -> None:
        df.to_csv(filename, index=False)

    @staticmethod
    def save_prompt(name: str, prompt: str) -> None:
        df = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        new_row = pd.DataFrame({
            'name': [name],
            'timestamp': [datetime.now()],
            'prompt': [prompt]
        })
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
                    st.experimental_rerun()
            with col2:
                if st.button("Delete", key=f"delete_{index}"):
                    df = df.drop(index)
                    DataManager.save_data(df, 'prompt_elements.csv')
                    st.success("Deleted successfully!")
                    st.experimental_rerun()

class PromptBuilder:
    @staticmethod
    def render():
        df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)

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

        recursive_feedback = st.checkbox("Request recursive feedback")

        prompt = PromptBuilder._generate_prompt(selections, df, recursive_feedback)
        PromptBuilder._display_prompt(prompt)

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
    def _generate_prompt(selections: Dict[str, Dict], df: pd.DataFrame,
                         recursive_feedback: bool) -> str:
        prompt_parts = []
        for section, data in selections.items():
            sel = data['selected']
            if (isinstance(sel, str) and sel == "Skip") or (isinstance(sel, list) and (not sel or sel == ["Skip"])):
                continue

            section_title = section.title()
            if section in ['audience', 'context', 'output']:
                if section == 'audience':
                    section_title = "Target Audience"
                if isinstance(sel, list):
                    if "Write your own" in sel and data['custom']:
                        content = data['custom']
                    else:
                        chosen = [s for s in sel if s not in ("Skip", "Write your own")]
                        content = "\n".join([df[df['title'] == a]['content'].values[0] for a in chosen if not df[df['title'] == a].empty])
                else:
                    content = data['custom'] if sel == "Write your own" else df[df['title'] == sel]['content'].values[0]
                if content:
                    prompt_parts.append(f"{section_title}:\n{content}")
            else:
                if isinstance(sel, str) and sel == "Write your own":
                    content = data['custom']
                else:
                    content = df[df['title'] == sel]['content'].values[0] if isinstance(sel, str) and not df[df['title'] == sel].empty else ""
                if content:
                    prompt_parts.append(f"{section_title}: {content}")

        prompt = "\n\n".join(prompt_parts)
        if recursive_feedback:
            prompt += (
                "\n\nBefore you provide the response, please ask me any questions that you feel could "
                "help you craft a better response. If you feel you have enough information to craft this response, "
                "please just provide it."
            )
        return prompt

    @staticmethod
    def _display_prompt(prompt: str):
        st.text_area("Generated Prompt", value=prompt, height=250, key="generated_prompt")
        st.info("To edit this prompt, click in and edit. To copy, use Ctrl-A, then copy.")
        col1, col2 = st.columns(2)
        with col1:
            prompt_name = st.text_input("Prompt Name")
        with col2:
            if st.button("Save Prompt"):
                if prompt_name:
                    DataManager.save_prompt(prompt_name, prompt)
                    st.success("Prompt saved successfully!")

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
        # Export elements
        elements_df = DataManager.load_data('prompt_elements.csv', CSV_COLUMNS)
        buf1 = io.StringIO()
        elements_df.to_csv(buf1, index=False)
        st.download_button(
            "Download prompt_elements.csv",
            data=buf1.getvalue(),
            file_name="prompt_elements.csv",
            mime="text/csv",
            key="dl_elements"
        )
        # Export history
        history_df = DataManager.load_data('prompt_history.csv', PROMPT_HISTORY_COLUMNS)
        buf2 = io.StringIO()
        history_df.to_csv(buf2, index=False)
        st.download_button(
            "Download prompt_history.csv",
            data=buf2.getvalue(),
            file_name="prompt_history.csv",
            mime="text/csv",
            key="dl_history"
        )

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
    st.set_page_config(layout="wide", page_title="KMo's Prompt Creation Tool")
    set_theme()
    st.title("KMo's Prompt Creation Tool")

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
