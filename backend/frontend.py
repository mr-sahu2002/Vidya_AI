import streamlit as st
import requests
import os
import time
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000/api/v1"  # Your FastAPI backend endpoint

st.set_page_config(
    page_title="Virtual Classroom Manager",
    page_icon="🏫",
    layout="wide"
)

def main():
    st.title("🏫 Virtual Classroom Manager")
    st.caption("Create classes, upload content, search educational materials, and get AI explanations!")

    # Add a new tab for AI Explanation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Create Class", 
        "Upload Content", 
        "Search Content", 
        "Explain Concept (AI RAG)", # New Tab
        "System Status"
    ])

    # =========================
    # CREATE CLASS TAB
    # =========================
    with tab1:
        st.subheader("Create New Virtual Class")
        teacher_id = st.text_input("Teacher ID", "teacher_123", key="class_teacher_id")
        class_name = st.text_input("Class Name", "Math Class 101", key="class_name")
        subject = st.text_input("Subject", "Mathematics", key="class_subject")
        
        if st.button("Create Virtual Class"):
            with st.spinner("Creating classroom..."):
                payload = {
                    "teacher_id": teacher_id,
                    "class_name": class_name,
                    "subject": subject
                }
                
                response = requests.post(
                    f"{BACKEND_URL}/create-class",
                    data=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("🎉 Classroom created successfully!")
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    col1.metric("Class ID", result["class_id"])
                    col2.metric("Collection", result["collection"])
                    
                    st.subheader("Next Steps:")
                    st.info(f"""
                    1. **Upload content** to this class using Class ID: `{result['class_id']}`
                    2. **Search content** using this Class ID
                    3. Manage content in the Upload and Search tabs
                    4. **Get AI explanations** for concepts in this class.
                    """)
                    
                    st.json(result)
                else:
                    st.error(f"Error creating class: {response.text}")

    # =========================
    # UPLOAD CONTENT TAB
    # =========================
    with tab2:
        st.subheader("Upload Educational Materials")
        class_id = st.text_input("Class ID for Upload", key="upload_class_id")
        difficulty = st.selectbox(
            "Difficulty Level", 
            ["basic", "intermediate", "advanced"], 
            key="upload_difficulty"
        )
        grade_level = st.selectbox(
            "Grade Level", 
            ["1-5", "6-8", "9-10", "11-12", "college", "unknown"], # Added unknown as option
            key="upload_grade"
        )
        language = st.selectbox(
            "Content Language", 
            ["english", "hindi", "kannada"],
            key="upload_language"
        )
        
        uploaded_file = st.file_uploader("Upload PDF or TXT File", type=["pdf", "txt"], key="upload_file") # Support txt
        
        if st.button("Process Document", key="upload_btn") and uploaded_file:
            if not class_id:
                st.error("Please enter a Class ID to upload content.")
                uploaded_file = None # Clear file if class_id is missing
            
            if uploaded_file:
                with st.spinner("Processing document... This may take a moment for large files."):
                    # Determine file extension
                    file_ext = uploaded_file.name.split('.')[-1]
                    temp_file = f"temp_{int(time.time())}.{file_ext}"
                    
                    # Save to temp file
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Prepare payload
                    files = {"file": open(temp_file, "rb")}
                    data = {
                        "class_id": class_id,
                        "difficulty": difficulty,
                        "grade_level": grade_level,
                        "language": language
                    }
                    
                    # Send to backend
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/upload", 
                            files=files, 
                            data=data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Uploaded {result['chunks_stored']} chunks from '{result['filename']}' to class '{result['class_id']}'!")
                            st.json(result)
                        else:
                            st.error(f"Error uploading file: {response.status_code} - {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Connection error: Is the backend server running?")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during upload: {e}")
                    finally:
                        # Clean up
                        if "file" in files: # Ensure file was opened
                            files["file"].close()
                        os.remove(temp_file)
            else:
                st.warning("Please upload a file to process.")

    # =========================
    # SEARCH CONTENT TAB
    # =========================
    with tab3:
        st.subheader("Search Educational Content")
        class_id = st.text_input("Class ID for Search", key="search_class_id")
        query = st.text_area("Question/Search Query", "Explain quadratic equations in simple terms", key="search_query")
        
        st.subheader("Filters (Optional)")
        filters_col1, filters_col2, filters_col3 = st.columns(3)
        
        with filters_col1:
            difficulty_filter = st.selectbox(
                "Difficulty", 
                ["any", "basic", "intermediate", "advanced"],
                key="search_diff"
            )
        with filters_col2:
            grade_filter = st.selectbox(
                "Grade Level", 
                ["any", "1-5", "6-8", "9-10", "11-12", "college", "unknown"],
                key="search_grade"
            )
        with filters_col3:
            content_type = st.selectbox(
                "Content Type (from metadata)", 
                ["any", "textbook", "worksheet", "lecture", "example", "definition", "concept"], # Added more types
                key="search_type"
            )
        
        if st.button("Search Knowledge Base", key="search_btn"):
            if not class_id or not query:
                st.error("Please enter both Class ID and a Query.")
                return

            with st.spinner("Searching educational content..."):
                # Prepare filters
                payload_filters = {}
                if difficulty_filter != "any":
                    payload_filters["difficulty"] = difficulty_filter
                if grade_filter != "any":
                    payload_filters["grade_level"] = grade_filter
                if content_type != "any":
                    payload_filters["content_type"] = content_type
                
                payload = {
                    "class_id": class_id,
                    "query": query,
                    **payload_filters # Unpack filters into payload
                }
                
                try:
                    response = requests.post(f"{BACKEND_URL}/search", data=payload)
                    
                    if response.status_code == 200:
                        results = response.json()
                        st.success(f"🔍 Found {len(results['results'])} relevant chunks for your search.")
                        
                        if results['results']:
                            for i, result in enumerate(results["results"]):
                                with st.expander(f"📝 Result {i+1} | Score: {result['score']:.3f} | File: {result['metadata'].get('filename', 'N/A')} "):
                                    st.write(f"**Chunk Text:**")
                                    st.markdown(f"```\n{result['text']}\n```")
                                    st.write(f"**Metadata:**")
                                    st.json(result['metadata'])
                        else:
                            st.info("No matching content found for your query and filters in this class.")
                    else:
                        st.error(f"Search failed: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection error: Is the backend server running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred during search: {e}")

    # =========================
    # EXPLAIN CONCEPT (AI RAG) TAB - NEW!
    # =========================
    with tab4:
        st.subheader("🧠 Get AI Explanation for a Concept")
        st.info("The AI will use retrieved information from your class content to generate a comprehensive explanation, guided by a Chain-of-Thought approach.")

        explain_class_id = st.text_input("Class ID for Explanation", key="explain_class_id", 
                                         help="Enter the ID of the class whose content the AI should use for explanation.")
        concept_to_explain = st.text_input("Concept to Explain", "Secularism", key="concept_to_explain",
                                           help="The main concept you want the AI to explain (e.g., 'Thermodynamics', 'Democracy').")
        
        st.markdown("---")
        st.subheader("Analogy & Context (Optional)")
        analogy_context = st.text_input("Analogy Context (p1)", "", 
                                        help="Suggest a real-world context for an analogy. E.g., 'cooking' for chemical reactions.", key="analogy_context")
        subject_context = st.text_input("Subject Context (p2)", "",
                                        help="Provide a specific subject if the class has mixed content, e.g., 'Physics' or 'History'. If left empty, AI will try to infer from class subject.", key="subject_context")
        user_additional_prompt = st.text_area("Additional Instructions (prompt)", "",
                                              help="Add any specific requirements for the explanation, e.g., 'Explain it to a 10-year-old' or 'Focus on its societal impact'.", key="user_additional_prompt")
        
        k_examples = st.slider("Number of examples to retrieve (k)", 1, 10, 5, key="k_examples_slider",
                               help="How many top relevant content chunks should the AI retrieve from the class knowledge base?")

        if st.button("Generate Explanation", key="generate_explanation_btn"):
            if not explain_class_id or not concept_to_explain:
                st.error("Please enter both a Class ID and the Concept to Explain.")
                return

            with st.spinner("Generating explanation with RAG and AI... This may take a moment."):
                payload = {
                    "class_id": explain_class_id,
                    "conceptToExplain": concept_to_explain,
                    "p1": analogy_context, # Mapped to analogy_context in backend
                    "p2": subject_context, # Mapped to subject_context in backend
                    "prompt": user_additional_prompt, # Mapped to user_prompt in backend
                    "k_examples": k_examples
                }
                
                try:
                    response = requests.post(f"{BACKEND_URL}/explain-concept", data=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✨ Explanation Generated!")
                        st.subheader(f"Explanation for: {result.get('concept', concept_to_explain)}")
                        st.markdown(result.get("explanation", "No explanation received."), unsafe_allow_html=True)
                        st.divider()
                        st.subheader("Raw API Response:")
                        st.json(result)
                    else:
                        st.error(f"Failed to generate explanation: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Connection error: Is the backend server running? (Check FastAPI logs)")
                except Exception as e:
                    st.error(f"An unexpected error occurred during explanation generation: {e}")


    # =========================
    # SYSTEM STATUS TAB
    # =========================
    with tab5:
        st.subheader("System Monitoring")
        
        if st.button("Check API Health", key="health_btn"):
            try:
                response = requests.get(f"{BACKEND_URL}/health")
                if response.status_code == 200:
                    st.success(f"✅ System Healthy | {response.json()['time']}")
                else:
                    st.error(f"❌ System Unavailable: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection error: Is the backend server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        
        st.divider()
        st.subheader("Class Statistics")
        
        stats_class_id = st.text_input("Class ID for Stats", key="stats_class_id")
        if st.button("Get Class Stats", key="stats_btn"):
            if not stats_class_id:
                st.warning("Please enter a Class ID to get statistics.")
                return

            try:
                response = requests.get(f"{BACKEND_URL}/class-stats/{stats_class_id}")
                if response.status_code == 200:
                    stats = response.json()
                    
                    col1_stats, col2_stats, col3_stats = st.columns(3)
                    col1_stats.metric("Total Chunks", stats.get("chunk_count", "N/A"))
                    col2_stats.metric("Collection Name", stats.get("collection", "N/A"))
                    col3_stats.metric("Vector Dimension", stats.get("vectors", {}).get("size", "N/A"))
                    
                    st.subheader("Detailed Statistics")
                    st.json(stats)
                elif response.status_code == 404:
                    st.warning(f"Class ID '{stats_class_id}' not found.")
                else:
                    st.error(f"Failed to get stats: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Connection error: Is the backend server running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()