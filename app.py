################################
# IMPORTS
################################
import os
import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype
)
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

################################
# FUNCTIONS
################################
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def get_column_topN_sorted(df,column_name, ascending=False, qty_vals=5):
    # given a dataframe and a column name, returns the sorted count of values
    return df[column_name].groupby(df[column_name]).count().sort_values(ascending=ascending).head(qty_vals)

def get_column_true_count(df, column_name):
    # given a dataframe and a column name, returns the count of True values
    # supports only boolean columns dtype
    try:
        return df[column_name][df[column_name]==True].count()
    except Exception:
        return 0

def pd_series_print_donut(pd_series, title="", showlegend=False):
    # given a Pandas series, plots a donut chart
    fig = go.Figure(data=[go.Pie(labels=pd_series.index.tolist(), values=pd_series.tolist(),
                             hole=.3, pull=[0.2, 0, 0, 0, 0])])
    fig.update_traces(hoverinfo='label+percent', textinfo='label+value', textfont_size=15,
                      marker=dict(colors=px.colors.sequential.RdBu))
    fig.update_layout(title_text=title,showlegend=showlegend)
    
    return fig

################################
# CODE
################################


# ========================================= CREATE DATAFRAME FROM CSV  =========================================

# Read dataframe

df = pd.read_csv(
    "WK2_Airbnb_Amsterdam_listings_proj_solution.csv", index_col=0
)

# set page layout as wide
st.set_page_config(layout="wide")


# ========================================= SET UP THE WEBPAGE SIDEBAR  =========================================

# --- SET SIDEBAR COMPOSITION
with st.sidebar:
    st.divider()
    st.markdown('<p class="small-font">Last Updated: 02/10/2023</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="small-font">Coding and template by Paolo Pozzoli</p>', unsafe_allow_html=True)

    img_pp = Image.open(os.getcwd() + "/pp.jpg")

    st.image(img_pp,
            caption='Follow me on LinkedIn - https://www.linkedin.com/in/paolo-pozzoli-9bb5a183/',
            width=200)


# ========================================= SET UP THE MAIN CONTENTS FOR EACH SELECTION  =========================================

# Display title and text
col1, mid, col2 = st.columns([5,1,20])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/2560px-Pandas_logo.svg.png", width=200)
with col2:
    st.title("Week 2 - Filter your Airbnb Listings dataframe!")
    st.write(
    """This app is based on this blog [here](https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/).
    Can you think of ways to extend it with visuals?
    """
)


# ========================================= CREATE 2 TABS TO DISPLAY DATA  ========================================================

tab1_intro, tab2_insights = st.tabs(["🏁 INTRO", "🔍 INSIGHTS"])

with tab1_intro:
    # FILTER THE DATAFRAME AS PER USER CHOICE
    filtered_df = filter_dataframe(df)

    # DISPLAY THE FILTERED DATAFRAME
    st.dataframe(filtered_df)


with tab2_insights:
    # HOST ACCEPTANCE RATE
    with st.expander("**HOST ACCEPTANCE RATE INSIGHTS**"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", filtered_df["host_acceptance_rate"].min())
        col2.metric("Mean", "{:.2f}".format(filtered_df["host_acceptance_rate"].mean()))
        col3.metric("Median", "{:.2f}".format(filtered_df["host_acceptance_rate"].median())) 
        col4.metric("Max", filtered_df["host_acceptance_rate"].max())

    # SUPERHOSTS
    st.metric(label="SUPERHOSTS", value=get_column_true_count(filtered_df,"host_is_superhost"))

    # TOP 5 NEIGHBOURHOODS
    sorted_neighbourhood = get_column_topN_sorted(filtered_df,"neighbourhood")
    # generate histogram plot
    fig_neighbourhood = px.histogram(sorted_neighbourhood, x=sorted_neighbourhood.index.tolist(), y=sorted_neighbourhood,
                    text_auto=True, labels={'x':'', 'y':''})
    # display plot in streamlit
    with st.expander("**NEIGHBOURHOODS INSIGHTS**"):
        st.plotly_chart(fig_neighbourhood, theme="streamlit")

    # TOP 4 ROOM TYPE
    sorted_roomtype = get_column_topN_sorted(filtered_df,"room_type",qty_vals=4)
    # generate histogram plot
    fig_neighbourhood = px.histogram(sorted_roomtype, x=sorted_roomtype.index.tolist(), y=sorted_roomtype,
                    text_auto=True, labels={'x':'', 'y':''})
    # display plot in streamlit
    with st.expander("**ROOM TYPE INSIGHTS**"):
        st.plotly_chart(fig_neighbourhood, theme="streamlit")

    # TOP 5 ACCOMMODATES
    sorted_accommodates = get_column_topN_sorted(filtered_df,"accommodates")
    # generate pie/donut plot
    fig_accommodates_donut = pd_series_print_donut(sorted_accommodates, "TOP 5 accommodates")
    # display plot in streamlit
    with st.expander("**ACCOMMODATE INSIGHTS**"):
        st.plotly_chart(fig_accommodates_donut)

    # TOP 5 BEDROOMS
    sorted_bedrooms = get_column_topN_sorted(filtered_df,"bedrooms")
    # generate pie/donut plot
    fig_bedrooms_donut = pd_series_print_donut(sorted_bedrooms, "TOP 5 bedrooms")
    # display plot in streamlit
    with st.expander("**BEDROOM INSIGHTS**"):
        st.plotly_chart(fig_bedrooms_donut)

    # TOP 5 BEDS
    sorted_beds = get_column_topN_sorted(filtered_df,"beds")
    # generate pie/donut plot
    fig_beds_donut = pd_series_print_donut(sorted_beds, "TOP 5 beds")
    # display plot in streamlit
    with st.expander("**BEDS INSIGHTS**"):
        st.plotly_chart(fig_beds_donut)

    # TOP 5 AMENITIES
    sorted_amenities = get_column_topN_sorted(filtered_df,"amenities")
    # generate pie/donut plot
    fig_amenities_donut = pd_series_print_donut(sorted_amenities, "TOP 5 amenities")
    # display plot in streamlit
    with st.expander("**AMENITIES INSIGHTS**"):
        st.plotly_chart(fig_amenities_donut)

    # HAS AVAILABILITY QTY
    st.metric(label="HAS AVAILABILITY", value=get_column_true_count(filtered_df,"has_availability"))

    # REVIEW SCORES RATING
    with st.expander("**REVIEW SCORES RATE INSIGHTS**"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", filtered_df["review_scores_rating"].min())
        col2.metric("Mean", "{:.2f}".format(filtered_df["review_scores_rating"].mean()))
        col3.metric("Median", "{:.2f}".format(filtered_df["review_scores_rating"].median())) 
        col4.metric("Max", filtered_df["review_scores_rating"].max())

    # INSTANT BOOKABLE QTY
    st.metric(label="INSTANT BOOKABLE", value=get_column_true_count(filtered_df,"instant_bookable"))

    # PRICE PER PERSON RATING
    with st.expander("**PRICE PER PERSON RATE INSIGHTS**"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", filtered_df["price_per_person"].min())
        col2.metric("Mean", "{:.2f}".format(filtered_df["price_per_person"].mean()))
        col3.metric("Median", "{:.2f}".format(filtered_df["price_per_person"].median())) 
        col4.metric("Max", filtered_df["price_per_person"].max())

    # AVAILABLE RATING
    with st.expander("**AVAILABLE**"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", filtered_df["available"].min())
        col2.metric("Mean", "{:.2f}".format(filtered_df["available"].mean()))
        col3.metric("Median", "{:.2f}".format(filtered_df["available"].median())) 
        col4.metric("Max", filtered_df["available"].max())
