import os
import pickle
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
st.set_page_config(
    page_title="Course Recommender Pro",
    page_icon="🎓",
    layout="wide"
)

# Cache data loading
@st.cache_data
def load_resources():
    try:
        courses = pickle.load(open('models/courses.pkl', 'rb'))
        similarity = pickle.load(open('models/similarity.pkl', 'rb'))
        return courses, similarity
    except FileNotFoundError as e:
        st.error(f"Critical data files missing: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Initialize application
def main():
    # Load data with progress indicator
    with st.spinner('🚀 Loading course database...'):
        courses_df, similarity_matrix = load_resources()
    
    # Header Section
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #1a73e8;'>Smart Course Finder</h1>
        <h3 style='color: #666;'>
            Discover 3,500+ Coursera Courses by Name, Description, or Skills
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Search Interface
    search_col, filter_col = st.columns([3, 1])
    
    with search_col:
        search_input = st.text_input(
            "🔍 Search courses:",
            placeholder="Enter course name, keywords, or skills..."
        )
    
    with filter_col:
        difficulty_filter = st.selectbox(
            "🗂 Filter by difficulty:",
            ['All'] + list(courses_df['Difficulty Level'].unique())
        )

    # Recommendation Function with Error Handling
    def get_recommendations(course_name):
        try:
            matches = courses_df[courses_df['course_name'].str.lower() == course_name.lower()]
            if matches.empty:
                return []
            index = matches.index[0]
            sim_scores = list(enumerate(similarity_matrix[index]))
            sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
            return [courses_df.iloc[i[0]].course_name for i in sorted_scores]
        except Exception as e:
            st.error(f"Recommendation error: {str(e)}")
            return []

    # Search Functionality
    def enhanced_search(query):
        query = query.strip().lower()
        return courses_df[
            courses_df['course_name'].str.lower().str.contains(query) |
            courses_df['Course Description'].str.lower().str.contains(query) |
            courses_df['Skills'].str.lower().str.contains(query)
        ]

    # Display Results
    if search_input:
        results = enhanced_search(search_input)
        
        if not results.empty:
            if difficulty_filter != 'All':
                results = results[results['Difficulty Level'] == difficulty_filter]
            
            st.success(f"📚 Found {len(results)} matching courses:")
            
            # Display Course Cards
            cols = st.columns(3)
            for idx, (_, row) in enumerate(results.iterrows()):
                with cols[idx % 3]:
                    with st.expander(f"📘 {row['course_name']}"):
                        st.markdown(f"""
                        **🏫 University:** {row['University']}  
                        **⭐ Rating:** {row['Course Rating']}/5  
                        **📈 Difficulty:** {row['Difficulty Level']}  
                        **🔧 Skills:** {row['Skills'][:100]}...  
                        🔗 <a href="{row['Course URL']}" target="_blank" rel="noopener noreferrer"><b>Go to Course</b></a>
                        """, unsafe_allow_html=True)


            
            # Recommendation Section
            if 'selected_course' in st.session_state:
                recommendations = get_recommendations(st.session_state.selected_course)
                if recommendations:
                    st.markdown("---")
                    st.subheader("🎯 Recommended Similar Courses")
                    for i, course in enumerate(recommendations, 1):
                        st.markdown(f"{i}. **{course}**")
        else:
            st.warning("No courses found matching your search. Try different keywords.")

    # else:
    #     # Show Popular Courses
    #     st.info("🌟 Top Rated Courses")
    #     popular_courses = courses_df.sort_values('Course Rating', ascending=False).head(6)
    #     cols = st.columns(3)
    #     for idx, (_, row) in enumerate(popular_courses.iterrows()):
    #         with cols[idx % 3]:
    #             st.markdown(f"""
    #             <div style='padding:1rem; margin:0.5rem; border-radius:8px; 
    #                       background:#f8f9fa; box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
    #                 <h4>{row['course_name']}</h4>
    #                 <p>🏫 {row['University']}</p>
    #                 <p>⭐ {row['Course Rating']}/5 | 📈 {row['Difficulty Level']}</p>
    #             </div>
    #             """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
