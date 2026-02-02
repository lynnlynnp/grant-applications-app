################################
# CONFIGURATION
################################

import streamlit as st
import pandas as pd
from typing import Tuple
import hashlib
import joblib
from io import BytesIO
from umap import UMAP
from hdbscan import HDBSCAN

from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space

# ---- FUNCTIONS ----

from src.extract_usage import extract_usage
from src.necessity_index import compute_necessity, index_scaler, qcut_labels
from src.column_detection import detect_freeform_col, detect_id_col, detect_school_type_col
from src.shortlist import shortlist_applications
from src.twinkl_originals import find_book_candidates
from src.preprocess_text import normalise_text 
import src.models.topic_modeling_pipeline as topic_modeling_pipeline
from src.px_charts import plot_histogram, plot_topic_countplot
from src.logger import get_logger
import src.pr_opps as PR_Analyser

logger = get_logger(__name__)

style_metric_cards(box_shadow=False, border_left_color='#E7F4FF',background_color='#E7F4FF', border_size_px=0, border_radius_px=6)

##################################
# CACHED PROCESSING FUNCTIONS
##################################

# -----------------------------------------------------------------------------
# Heavy processing (IO + NLP) is cached to avoid re‑executing when the UI state
# changes. The function only re‑runs if the **file contents** change.
# -----------------------------------------------------------------------------

def load_heartfelt_predictor():
    model_path = "src/models/heartfelt_pipeline.joblib"
    return joblib.load(model_path)

@st.cache_resource
def load_embeddings_model():
    return topic_modeling_pipeline.load_embedding_model('all-MiniLM-L12-v2')

@st.cache_data(show_spinner=True)
def load_and_process(raw_csv: bytes) -> Tuple[pd.DataFrame, str]:
    """
    Load CSV from raw bytes, detect freeform column, compute necessity scores,
    and extract usage items. Returns processed DataFrame and freeform column name.
    """
    # Read Uploaded Data 
    df_orig = pd.read_csv(BytesIO(raw_csv))
    
    # Drop completely empty rows or junk rows with mostly NaNs
    df_orig = df_orig.dropna(how='all')
    if not df_orig.empty:
        df_orig = df_orig.dropna(thresh=df_orig.shape[1] // 2)

    df_orig.columns = df_orig.columns.str.lower().str.replace(' ', '_')

    # Detect freeform column
    freeform_col = detect_freeform_col(df_orig)
    id_col = detect_id_col(df_orig)
    school_type_col = detect_school_type_col(df_orig)
    
    # Fallbacks and validations
    if freeform_col is None:
        # If detection fails, try picking the column with the longest average string length
        obj_cols = df_orig.select_dtypes(include=["object"]).columns
        if not obj_cols.empty:
            freeform_col = df_orig[obj_cols].apply(lambda x: x.astype(str).str.len().mean()).idxmax()
        else:
            raise ValueError("Could not detect a text column for analysis. Please ensure your CSV contains application text.")

    if id_col is None:
        # Fallback to index if no ID detected
        df_orig['generated_id'] = range(1, len(df_orig) + 1)
        id_col = 'generated_id'
        
    if school_type_col is None:
        # Use a dummy column if no school type detected to avoid downstream errors
        df_orig['unknown_school_type'] = 'Unknown'
        school_type_col = 'unknown_school_type'

    logger.info(f"ID Column: {id_col}")
    logger.info(f"Freeform Column: {freeform_col}")

    df_orig = df_orig[df_orig[freeform_col].notna()]
    df_orig = df_orig.assign(**{id_col: df_orig[id_col].astype(int)})

    #Word Count
    df_orig['word_count'] = df_orig[freeform_col].fillna('').str.split().str.len()

    # Compute Necessity Scores
    scored = df_orig.join(df_orig[freeform_col].apply(compute_necessity))
    scored['necessity_index'] = index_scaler(scored['necessity_index'].values)
    scored['priority'] = qcut_labels(scored['necessity_index'])

    # Find Twinkl Originals Candidates
    scored['book_candidates'] = find_book_candidates(scored, freeform_col, school_type_col)

    # Label Heartfelt Applications
    scored['clean_text'] = scored[freeform_col].map(normalise_text)
    model = load_heartfelt_predictor()
    scored['is_heartfelt'] = model.predict(scored['clean_text'].astype(str))


    
    # Usage Extraction
    docs = df_orig[freeform_col].to_list()
    scored['usage'] = [tuple(l) for l in extract_usage(docs)]

    return scored, freeform_col, id_col

# -----------------------------------------------------------------------------
# Derivative computations that rely only on the processed DataFrame are also
# cached. These are lightweight but still benefit from caching because this
# function might be called multiple times during widget interaction.
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def compute_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """Pre‑compute shortlist_score for all rows (used for both modes)."""
    return shortlist_applications(df, k=len(df))

@st.cache_data(show_spinner=True)
def analyse_pr_opps(
        shortlisted_apps_df: pd.DataFrame,
        freeform_col: str,
        id_col: str
        ) -> pd.DataFrame:

    return PR_Analyser.analyse(shortlisted_apps_df, freeform_col, id_col)


@st.cache_data
def apply_filters(df: pd.DataFrame, auto_shortlist_df: pd.DataFrame, 
                  selected_view: str, filter_range: tuple, selected_topics: list) -> pd.DataFrame:
    """
    Cache filtering operations to avoid recomputing on every interaction.
    """
    # Define filter functions
    def filter_all_applications():
        return df[df['necessity_index'].between(filter_range[0], filter_range[1])]

    def filter_not_shortlisted():
        return df[
            (~df.index.isin(auto_shortlist_df.index)) &
            (df['necessity_index'].between(filter_range[0], filter_range[1]))
        ]

    filter_map = {
        'All applications': filter_all_applications,
        'Not shortlisted': filter_not_shortlisted,
    }

    filtered_df = filter_map[selected_view]()
    
    # Apply topic filtering if topics are selected
    if selected_topics:
        selected_set = set(selected_topics)
        filtered_df = filtered_df[
            filtered_df['topics'].apply(lambda topic_list: selected_set.issubset(set(topic_list)))
        ]
    
    return filtered_df

@st.cache_data
def prepare_display_data(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process display data once to avoid repeated computations.
    """
    display_df = filtered_df.copy()
    display_df['clean_usage'] = display_df['usage'].apply(
        lambda x: [item for item in x if item and item.lower() != 'none']
    )
    display_df['clean_topics'] = display_df['topics'].apply(
        lambda x: [item for item in x if item and item.lower() != 'none']
    )
    return display_df

@st.cache_resource(show_spinner=True)
def run_topic_modeling(_df: pd.DataFrame, freeform_col: str, _file_hash: str):
    try:
        # ------- 1. Tokenize texts into sentences -------
        nlp = topic_modeling_pipeline.load_spacy_model(model_name='en_core_web_sm')

        sentences = []
        mappings = []

        for idx, application_text in _df[freeform_col].dropna().items():
            for sentence in topic_modeling_pipeline.spacy_sent_tokenize(application_text):
                sentences.append(sentence)
                mappings.append(idx)

        # ✅ Check if we have enough data
        if len(sentences) < 20:
            st.warning(f"⚠️ Only {len(sentences)} sentences found. Topic modeling requires at least 20 sentences for meaningful results.")
            return None, None, None, None

        # -------- 2. Generate embeddings -------
        embeddings_model = load_embeddings_model()
        embeddings = embeddings_model.encode(sentences, show_progress_bar=True)

        # -------- 3. Dynamic Topic Modeling Parameters --------
        # Adjust min_cluster_size based on data size
        min_cluster_size = max(5, min(10, len(sentences) // 10))
        
        umap_model = UMAP(
            n_neighbors=min(7, len(sentences) - 1),  # Can't be larger than n_samples - 1
            n_components=3, 
            min_dist=0.0, 
            metric='cosine', 
            random_state=42
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )

        # --------- 4. Perform Topic Modeling ---------
        topic_model, topics, probs = topic_modeling_pipeline.bertopic_model(
            sentences, embeddings, embeddings_model, umap_model, hdbscan_model
        )

        topic_modeling_pipeline.ai_labels_to_custom_name(topic_model) 

        return topic_model, topics, probs, mappings

    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        raise

################################
# MAIN APP SCRIPT
################################

st.title("🪷 Community Collections Helper")

uploaded_file = st.file_uploader("Upload grant applications file for analysis", type='csv', label_visibility='collapsed')


# ========== FINGERPRINTING CURRENT FILE ==========
# This helps avoid reruns of certain functions as
# long as the file stays the same

if uploaded_file is not None:
    raw = uploaded_file.read()
    file_hash = hashlib.md5(raw).hexdigest()
    st.session_state["current_file_hash"] = file_hash
else:
    raw = None
    st.session_state.pop("current_file_hash", None)

if raw is None:
    st.stop()

## ============== DATA PROCESSING ================

try:
    logger.info("🧪 APPLICATION ANALYSIS STARTED")
    df, freeform_col, id_col = load_and_process(raw) # from cached function
    file_hash = st.session_state["current_file_hash"]
    logger.info(f"Preprocessing successful. Identified {freeform_col} as freeform column and {id_col} as application ID column")
    
except Exception as e:
    logger.error(f"Error preprocessing file: {str(e)}")

try:
    logger.info("Topic Modeling initiated")
    topic_model, topics, probs, mappings = run_topic_modeling(df, freeform_col, file_hash)

    label_map = (topic_model
                 .get_topic_info()
                 .set_index("Topic")["CustomName"]
                 .to_dict())

    df = topic_modeling_pipeline.attach_topics(df, mappings, topics, label_map, col="topics")

    topics_df = (topic_model.get_topic_info()
                 .query('Topic > -1')
                 .drop(columns = ['Name', 'OpenAI'])
                 .rename(columns = {'CustomName':'Topic Name', 'Topic':'Topic Nr.'})
                 )

    cols_to_move = ['Topic Nr.', 'Topic Name']
    topics_df = topics_df[cols_to_move + [col for col in topics_df.columns if col not in cols_to_move]]

except Exception as e:
    st.error("⚠️  Topic modeling failed. Please check the error message above and try again.")
    st.stop()

book_candidates_df = df[df['book_candidates'] == True]

###############################
#         SIDE PANNEL         #
###############################

with st.sidebar:
    st.title("Shortlist Mode")


    quantile_map = {"strict": 0.75, "generous": 0.5}
    mode = st.segmented_control(
            "Select one option",
            options=["strict", "generous"],
            default="strict",
            )
    
    scored_full = compute_shortlist(df)
    threshold_score = scored_full["shortlist_score"].quantile(quantile_map[mode])
    auto_shortlist_df = scored_full[scored_full["shortlist_score"] >= threshold_score]

    st.title("Filters")

    ## --- Dataframe To Filter ---
    options = ['All applications', 'Not shortlisted'] 
    selected_view = st.pills('Choose data to filter', options, default='Not shortlisted')
    st.write("")

    ## --- Necessity Index Filtering ---
    min_idx = float(df['necessity_index'].min())
    max_idx = float(df['necessity_index'].max())
    filter_range = st.slider(
        "Necessity Index Range", min_value=min_idx, max_value=max_idx, value=(min_idx, max_idx)
    )

    ## -------- Topic Filtering -------

    topic_options = sorted(topics_df['Topic Name'].unique())
    selected_topics = st.multiselect("Filter by Topic(s)", options=topic_options, default=[])

    # Use cached filtering function
    filtered_df = apply_filters(df, auto_shortlist_df, selected_view, filter_range, selected_topics)

    st.markdown(f"**Total Applications:** {len(df)}")
    st.markdown(f"**Filtered Applications:** {len(filtered_df)}")

    manual_keys = [k for k in st.session_state.keys() if k.startswith("shortlist_")]
    manually_shortlisted = [int(k.split("_")[1]) for k in manual_keys if st.session_state[k]]

    st.markdown(f"**Manually Shortlisted:** {len(manually_shortlisted)}")
    if manually_shortlisted:
        csv = df.loc[manually_shortlisted].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Manual Shortlist",
            data=csv,
            file_name="manual_shortlist.csv",
            mime="text/csv",
            icon="⬇️",
        )


    add_vertical_space(4)
    st.divider()
    st.badge("Version 1.3.0", icon=':material/category:',color='violet')
    st.markdown("""
    :grey[Made with 🩷  by the AI Innovation Team
    Contact: lynn.perez@twinkl.com]
    """)



## ====== CREATE TAB SECTIONS =======
tab1, tab2, tab3 = st.tabs(["Shortlist Manager","Insights", "PR Opportunities"])


##################################################
#              SHORTLIST MANAGER TAB             # 
##################################################

with tab1:
    
    ## =========== AUTOMATIC SHORTLIST =========

    st.header("Automatic Shortlist")

    csv_auto = auto_shortlist_df.to_csv(index=False).encode("utf-8")
    all_processed_data = df.to_csv(index=False).encode("utf-8")
    book_candidates = book_candidates_df.to_csv(index=False).encode("utf-8")
    topic_descriptions_csv = topics_df.to_csv(index=False).encode("utf-8")


    csv_options = {
        "Shortlist": (csv_auto, "shortlist.csv"),
        "All Processed Data": (all_processed_data, "all_processed.csv"),
        "Book Candidates": (book_candidates, "book_candidates.csv"),
        "Topic Descriptions": (topic_descriptions_csv, "topic_descriptions.csv"),
    }

    choice = st.selectbox("Select a file for download", list(csv_options.keys()))

    csv_data, file_name = csv_options[choice]


    st.download_button(
        label=f"Download {choice}",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        help="This button will download the selected file from above",
        icon="⬇️"

    )

    
    st.write("")
    total_col, shortlistCounter_col, mode_col = st.columns(3)

    total_col.metric("Applications Submitted", len(df))
    shortlistCounter_col.metric("Shorlist Length",  len(auto_shortlist_df))
    mode_col.metric("Mode", mode)

    shorltist_cols_to_show = [
            id_col,
            freeform_col,
            'book_candidates',
            'usage',
            'necessity_index',
            'urgency_score',
            'severity_score',
            'vulnerability_score',
            'shortlist_score',
            'is_heartfelt',
            'topics',
            ]

    st.dataframe(auto_shortlist_df.loc[:, shorltist_cols_to_show], hide_index=True)

    ## ====== APPLICATIONS REVIEW =======

    add_vertical_space(2)
    st.header("Manual Filtering")
    st.info("Use the **side panel** filters to more easily sort through applications that you'd like to review.", icon=':material/info:')

    st.write("")
    if len(filtered_df) > 0:
        st.markdown("#### Filtered Applications")

        # Pre-process display data using cached function
        display_df = prepare_display_data(filtered_df)

        # ========== PAGINATION IMPLEMENTATION ==========
        items_per_page = 10
        total_items = len(display_df)
        total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1

        # Initialize page number in session state if not present
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        # Calculate start and end indices for current page
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        st.write("")
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_items} applications")

        # Display only items for current page
        for row in display_df.iloc[start_idx:end_idx].itertuples():
            with st.expander(f"Application {int(getattr(row, id_col))}"):
                st.write("")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Necessity", f"{row.necessity_index:.1f}")
                col2.metric("Urgency", f"{int(row.urgency_score)}")
                col3.metric("Severity", f"{int(row.severity_score)}")
                col4.metric("Vulnerability", f"{int(row.vulnerability_score)}")

                st.markdown("##### Excerpt")
                st.write(getattr(row, freeform_col))


                # HTML for clean usage items 
                usage_items = row.clean_usage
                if usage_items:
                    st.markdown("##### Usage")
                    pills_html = "".join(
                            f"<span style='display:inline-block;background-color:#E7F4FF;color:#125E9E;border-radius:20px;padding:4px 10px;margin:2px;font-size:0.95rem;'>{item}</span>"
                        for item in usage_items
                    )
                    st.html(pills_html)
                else:
                    st.caption("*No usage found*")

                topic_items = row.clean_topics
                if topic_items:
                    st.markdown("##### Topics")
                    topic_boxes_html= "".join(
                    f"<span style='display:inline-block;background-color:#ECE0FC;color:#6741B9;border-radius:5px;padding:4px 10px;margin:2px;font-size:0.95rem;'>{item}</span>"
                    for item in topic_items
                    )
                    st.html(topic_boxes_html)
                else:
                    st.caption("_No topics assigned for this application_")


                st.checkbox(
                    "Add to shortlist",
                    key=f"shortlist_{row.Index}"
                )

        # PAGINATION CONTROLS

        st.write("")

        col_left, col_center, col_right = st.columns([2, 1, 2])
        
        with col_left:
            if st.button("⬅️  Previous", disabled=(st.session_state.current_page == 1), use_container_width=True):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col_center:
            st.markdown(f"<div style='text-align: center'>Page {st.session_state.current_page} of {total_pages}</div>", 
                       unsafe_allow_html=True)
        
        with col_right:
            if st.button("Next ➡️", disabled=(st.session_state.current_page == total_pages), use_container_width=True):
                st.session_state.current_page += 1
                st.rerun()

    else:
        st.markdown(
            """
            <br>
            <div style="text-align: center; font-size: 1.2em">
                🍂 <span style="color: grey;">No applications matched these filters...</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


#########################################
#              INSIGHTS TAB             #
#########################################

with tab2:


    ## =========== DATA OVERVIEW ==========

    st.header("General Insights")
    add_vertical_space(1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Applications Submitted", len(df))
    col2.metric("Median N.I", df['necessity_index'].median().round(2))
    col3.metric("Avg. Word Count", f"{df['word_count'].mean().round(1)}")

    ## --- NI Distribution Plot ---
    ni_distribution_plt = plot_histogram(df, col_to_plot='necessity_index', bins=50, title='Necessity Index Histogram')
    st.plotly_chart(ni_distribution_plt)


    ## ============= TOPIC MODELING ============

    st.header("Topic Modeling")
    add_vertical_space(1)
    
    ## ------- Display Topics Dataframe ------

    with st.popover("How are topic extracted?", icon="🌱"):

        st.write("""
        **About Topic Modeling**

        We use BERTopic to :primary[**dynamically**] extract the most common topics from the natural language data.

        BERTopic is a machine learning technique that allows us to group documents (in this case, sentences within application letters) based on their semantic similarity and other patterns such as word frequency and placement.

        The table you see below shows you the extracted topics, alongside their top 10 extracted keywords and a small sample of real texts from the applications that demonstrate where the topics came from.

        **Table Info**
        - **Topic Nr.:** The 'id' of the topic.
        - **Topic Name:** This is an AI-generated label based on a few samples of application responses alongside their corresponding keywords.
        - **Representation:** Top 10 keywords that best represent a topic
        - **Representative Docs**: Sample sentences contributing to the topic
        """)
    st.dataframe(topics_df, hide_index=True)

    ## -------- Plot Topics Chart ----------

    topic_count_plot = plot_topic_countplot(topics_df, topic_id_col='Topic Nr.', topic_name_col='Topic Name', representation_col='Representation', height=500, title='Topic Frequency Chart')
    st.plotly_chart(topic_count_plot, use_container_width=True)

    ## --------- User Updates -----------

    if st.session_state.get("topic_toast_shown_for") != st.session_state["current_file_hash"]:
        st.toast(
        """
        **Topic modeling is ready!** View the results on the _Insights_ tab
        """,
        icon='🎉'
        )

        st.session_state["topic_toast_shown_for"] = st.session_state["current_file_hash"]


#################################
#   PR OPPORTUNITIES TAB
#################################

with tab3:

    st.header("PR Opportunity Analysis")

    # ------------ PROCESSING ------------

    analysed_pr_apps_df = analyse_pr_opps(auto_shortlist_df, freeform_col, id_col)

    pr_opps_csv = analysed_pr_apps_df.to_csv(index=False).encode('utf-8')

    st.download_button(
            "Download PR Analysis",
            data=pr_opps_csv,
            file_name="pr_opportunities.csv",
            mime="text/csv",
            icon="⬇️",
    )

    st.dataframe(
            analysed_pr_apps_df.loc[analysed_pr_apps_df['fit_for_pr']].drop(columns='fit_for_pr'),
            hide_index=True
    )

