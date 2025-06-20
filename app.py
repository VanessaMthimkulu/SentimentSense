import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime
import os

## Default to 8501 
PORT = os.getenv("PORT", 8501)  
st.set_page_config(page_title="Sentiment Analysis App")

# Import utility modules
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.text_processor import TextProcessor
from utils.visualizations import create_sentiment_chart, create_confidence_chart, create_comparison_chart
from utils.file_handler import FileHandler
from utils.export_utils import ExportUtils

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if 'results' not in st.session_state:
    st.session_state.results = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

def main():
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("Analyze emotional tone in text data with confidence scoring and interactive visualizations")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Single Text Analysis", "Batch Processing", "Comparative Analysis", "Results Dashboard"]
    )
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.info("""
    **Model**: Rule-based Sentiment Analyzer
    
    **Classes**: Positive, Negative, Neutral
    
    **Confidence Threshold**: 0.5
    
    **Limitations**: 
    - Best for English text
    - Max 5000 characters per text
    - Uses lexicon-based approach
    """)
    
    if page == "Single Text Analysis":
        single_text_analysis()
    elif page == "Batch Processing":
        batch_processing()
    elif page == "Comparative Analysis":
        comparative_analysis()
    elif page == "Results Dashboard":
        results_dashboard()

def single_text_analysis():
    st.header("Single Text Analysis")
    
    # Text input options
    input_method = st.radio("Choose input method:", ["Direct Text Entry", "File Upload"])
    
    text_to_analyze = ""
    
    if input_method == "Direct Text Entry":
        text_to_analyze = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt', 'csv'],
            help="Upload a .txt file or .csv file with text content"
        )
        
        if uploaded_file is not None:
            file_handler = FileHandler()
            try:
                if uploaded_file.type == "text/plain":
                    text_to_analyze = file_handler.read_text_file(uploaded_file)
                else:
                    df = file_handler.read_csv_file(uploaded_file)
                    st.write("CSV Preview:")
                    st.dataframe(df.head())
                    
                    # Let user select text column
                    text_columns = df.select_dtypes(include=['object']).columns.tolist()
                    if text_columns:
                        selected_column = st.selectbox("Select text column:", text_columns)
                        # Combine all text from selected column
                        text_to_analyze = " ".join(df[selected_column].astype(str).tolist())
                    else:
                        st.error("No text columns found in the CSV file.")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    if text_to_analyze and st.button("Analyze Sentiment", type="primary"):
        with st.spinner("Analyzing sentiment..."):
            try:
                # Perform sentiment analysis
                result = st.session_state.analyzer.analyze_sentiment(text_to_analyze)
                
                # Process text for keywords
                text_processor = TextProcessor()
                keywords = text_processor.extract_keywords(text_to_analyze, num_keywords=10)
                
                # Store result
                analysis_result = {
                    'text': text_to_analyze,
                    'sentiment': result['label'],
                    'confidence': result['score'],
                    'keywords': keywords,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.current_analysis = analysis_result
                st.session_state.results.append(analysis_result)
                
                # Display results
                display_single_analysis_results(analysis_result)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

def display_single_analysis_results(result):
    st.success("Analysis Complete!")
    
    # Main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_color = {
            'POSITIVE': 'green',
            'NEGATIVE': 'red',
            'NEUTRAL': 'orange'
        }
        st.metric(
            "Sentiment", 
            result['sentiment'],
            help=f"Confidence: {result['confidence']:.2%}"
        )
    
    with col2:
        st.metric(
            "Confidence Score", 
            f"{result['confidence']:.2%}",
            help="Model's confidence in the prediction"
        )
    
    with col3:
        confidence_level = "High" if result['confidence'] > 0.8 else "Medium" if result['confidence'] > 0.5 else "Low"
        st.metric("Confidence Level", confidence_level)
    
    # Visualizations
    st.subheader("Sentiment Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence chart
        fig_conf = create_confidence_chart(result['confidence'], result['sentiment'])
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Sentiment distribution (single result)
        sentiment_data = pd.DataFrame([{
            'Sentiment': result['sentiment'],
            'Confidence': result['confidence'],
            'Count': 1
        }])
        fig_sent = create_sentiment_chart(sentiment_data)
        st.plotly_chart(fig_sent, use_container_width=True)
    
    # Keywords section
    st.subheader("Key Sentiment Drivers")
    if result['keywords']:
        # Create keyword tags
        keyword_html = ""
        for keyword in result['keywords']:
            keyword_html += f'<span style="background-color: #e1f5fe; padding: 2px 6px; margin: 2px; border-radius: 3px; display: inline-block;">{keyword}</span> '
        st.markdown(keyword_html, unsafe_allow_html=True)
    else:
        st.info("No significant keywords extracted.")
    
    # Text with highlighted keywords
    st.subheader("Text Analysis")
    highlighted_text = highlight_keywords(result['text'], result['keywords'])
    st.markdown(highlighted_text, unsafe_allow_html=True)
    
    # Export options
    import streamlit as st
from datetime import datetime

# Dummy placeholder for your analysis function
def analyze_sentiment(text):
    return {"text": text, "sentiment": "Positive", "score": 0.8}

# Placeholder for export helpers
class ExportUtils:
    def export_to_json(self, data):
        import json
        return json.dumps(data, indent=2)

    def export_to_csv(self, data):
        import pandas as pd
        from io import StringIO
        df = pd.DataFrame(data)
        output = StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()

# UI starts here
st.title("🚀 Sentiment Analysis")

user_input = st.text_area("Enter text for analysis")

if st.button("Analyze"):
    result = analyze_sentiment(user_input)
    st.session_state["result"] = result  # Save result for export
    st.write(f"**Sentiment:** {result['sentiment']}")
    st.write(f"**Confidence Score:** {result['score']}")

# Show export options only if there's something to export
if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Export Results")
    export_utils = ExportUtils()

    col1, col2 = st.columns(2)

    with col1:
        json_data = export_utils.export_to_json([result])
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        csv_data = export_utils.export_to_csv([result])
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def batch_processing():
    st.header("Batch Processing")
    st.markdown("Upload multiple texts for batch sentiment analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with text data",
        type=['csv'],
        help="Upload a CSV file with a column containing text to analyze"
    )
    
    if uploaded_file is not None:
        try:
            file_handler = FileHandler()
            df = file_handler.read_csv_file(uploaded_file)
            
            st.write("File Preview:")
            st.dataframe(df.head())
            
            # Column selection
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if not text_columns:
                st.error("No text columns found in the CSV file.")
                return
            
            selected_column = st.selectbox("Select text column for analysis:", text_columns)
            
            # Optional: ID column
            id_columns = df.columns.tolist()
            id_column = st.selectbox(
                "Select ID column (optional):", 
                ["None"] + id_columns,
                help="Column to use as identifier for each text"
            )
            
            # Processing options
            st.subheader("Processing Options")
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.slider("Batch size", 1, 50, 10, help="Number of texts to process at once")
            
            with col2:
                extract_keywords = st.checkbox("Extract keywords", value=True)
            
            if st.button("Start Batch Processing", type="primary"):
                with st.spinner("Processing batch..."):
                    try:
                        # Prepare data
                        texts = df[selected_column].astype(str).tolist()
                        ids = df[id_column].tolist() if id_column != "None" else list(range(len(texts)))
                        
                        # Initialize progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        batch_results = []
                        text_processor = TextProcessor() if extract_keywords else None
                        
                        # Process in batches
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i+batch_size]
                            batch_ids = ids[i:i+batch_size]
                            
                            for j, (text, text_id) in enumerate(zip(batch_texts, batch_ids)):
                                # Update progress
                                current_progress = (i + j + 1) / len(texts)
                                progress_bar.progress(current_progress)
                                status_text.text(f"Processing {i + j + 1}/{len(texts)}: {str(text_id)}")
                                
                                # Analyze sentiment
                                result = st.session_state.analyzer.analyze_sentiment(text)
                                
                                # Extract keywords if requested
                                keywords = []
                                if extract_keywords and text_processor:
                                    keywords = text_processor.extract_keywords(text, num_keywords=5)
                                
                                batch_result = {
                                    'id': text_id,
                                    'text': text[:200] + "..." if len(text) > 200 else text,  # Truncate for display
                                    'full_text': text,
                                    'sentiment': result['label'],
                                    'confidence': result['score'],
                                    'keywords': keywords,
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                batch_results.append(batch_result)
                        
                        # Store results
                        st.session_state.results.extend(batch_results)
                        
                        # Display results
                        display_batch_results(batch_results)
                        
                        progress_bar.progress(1.0)
                        status_text.text("Batch processing complete!")
                        
                    except Exception as e:
                        st.error(f"Error during batch processing: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def display_batch_results(results):
    st.success(f"Batch processing complete! Analyzed {len(results)} texts.")
    
    # Convert to DataFrame for easier manipulation
    df_results = pd.DataFrame(results)
    
    # Summary statistics
    st.subheader("Batch Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Texts", len(results))
    
    with col2:
        avg_confidence = df_results['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    with col3:
        positive_count = len(df_results[df_results['sentiment'] == 'POSITIVE'])
        st.metric("Positive Texts", positive_count)
    
    with col4:
        negative_count = len(df_results[df_results['sentiment'] == 'NEGATIVE'])
        st.metric("Negative Texts", negative_count)
    
    # Visualizations
    st.subheader("Batch Results Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        sentiment_counts = df_results['sentiment'].value_counts()
        fig_dist = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEGATIVE': '#e74c3c',
                'NEUTRAL': '#f39c12'
            }
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig_conf = px.histogram(
            df_results,
            x='confidence',
            nbins=20,
            title="Confidence Score Distribution",
            labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        sentiment_filter = st.multiselect(
            "Filter by sentiment:",
            options=df_results['sentiment'].unique(),
            default=df_results['sentiment'].unique()
        )
    
    with col2:
        min_confidence = st.slider("Minimum confidence:", 0.0, 1.0, 0.0, 0.1)
    
    # Apply filters
    filtered_df = df_results[
        (df_results['sentiment'].isin(sentiment_filter)) &
        (df_results['confidence'] >= min_confidence)
    ]
    
    # Display table
    display_columns = ['id', 'text', 'sentiment', 'confidence']
    if any(result['keywords'] for result in results):
        display_columns.append('keywords')
    
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Export options
    st.subheader("Export Batch Results")
    export_utils = ExportUtils()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Batch as JSON"):
            json_data = export_utils.export_to_json(results)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"batch_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export Batch as CSV"):
            csv_data = export_utils.export_to_csv(results)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"batch_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def comparative_analysis():
    st.header("Comparative Analysis")
    st.markdown("Compare sentiment analysis results between different text sources")
    
    # Text input for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text Source A")
        text_a = st.text_area("Enter first text:", height=150, key="text_a")
        label_a = st.text_input("Label for Text A:", value="Text A")
    
    with col2:
        st.subheader("Text Source B")
        text_b = st.text_area("Enter second text:", height=150, key="text_b")
        label_b = st.text_input("Label for Text B:", value="Text B")
    
    if st.button("Compare Texts", type="primary") and text_a and text_b:
        with st.spinner("Analyzing both texts..."):
            try:
                # Analyze both texts
                result_a = st.session_state.analyzer.analyze_sentiment(text_a)
                result_b = st.session_state.analyzer.analyze_sentiment(text_b)
                
                # Extract keywords
                text_processor = TextProcessor()
                keywords_a = text_processor.extract_keywords(text_a, num_keywords=8)
                keywords_b = text_processor.extract_keywords(text_b, num_keywords=8)
                
                # Display comparison
                display_comparison_results(
                    {
                        'text': text_a,
                        'label': label_a,
                        'sentiment': result_a['label'],
                        'confidence': result_a['score'],
                        'keywords': keywords_a
                    },
                    {
                        'text': text_b,
                        'label': label_b,
                        'sentiment': result_b['label'],
                        'confidence': result_b['score'],
                        'keywords': keywords_b
                    }
                )
                
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")

def display_comparison_results(result_a, result_b):
    st.success("Comparison Complete!")
    
    # Side-by-side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📝 {result_a['label']}")
        st.metric("Sentiment", result_a['sentiment'])
        st.metric("Confidence", f"{result_a['confidence']:.2%}")
        
        if result_a['keywords']:
            st.write("**Keywords:**")
            keyword_html = ""
            for keyword in result_a['keywords']:
                keyword_html += f'<span style="background-color: #e3f2fd; padding: 2px 6px; margin: 2px; border-radius: 3px; display: inline-block;">{keyword}</span> '
            st.markdown(keyword_html, unsafe_allow_html=True)
    
    with col2:
        st.subheader(f"📝 {result_b['label']}")
        st.metric("Sentiment", result_b['sentiment'])
        st.metric("Confidence", f"{result_b['confidence']:.2%}")
        
        if result_b['keywords']:
            st.write("**Keywords:**")
            keyword_html = ""
            for keyword in result_b['keywords']:
                keyword_html += f'<span style="background-color: #fff3e0; padding: 2px 6px; margin: 2px; border-radius: 3px; display: inline-block;">{keyword}</span> '
            st.markdown(keyword_html, unsafe_allow_html=True)
    
    # Comparison visualization
    st.subheader("Comparison Visualization")
    
    comparison_data = pd.DataFrame([
        {'Source': result_a['label'], 'Sentiment': result_a['sentiment'], 'Confidence': result_a['confidence']},
        {'Source': result_b['label'], 'Sentiment': result_b['sentiment'], 'Confidence': result_b['confidence']}
    ])
    
    fig_comparison = create_comparison_chart(comparison_data)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Analysis insights
    st.subheader("Comparison Insights")
    
    # Sentiment comparison
    if result_a['sentiment'] == result_b['sentiment']:
        st.info(f"✅ Both texts have the same sentiment: **{result_a['sentiment']}**")
    else:
        st.warning(f"⚠️ Different sentiments detected: **{result_a['label']}** is {result_a['sentiment']} while **{result_b['label']}** is {result_b['sentiment']}")
    
    # Confidence comparison
    conf_diff = abs(result_a['confidence'] - result_b['confidence'])
    if conf_diff < 0.1:
        st.info(f"📊 Similar confidence levels (difference: {conf_diff:.2%})")
    else:
        higher_conf = result_a['label'] if result_a['confidence'] > result_b['confidence'] else result_b['label']
        st.info(f"📊 **{higher_conf}** has higher confidence (difference: {conf_diff:.2%})")
    
    # Keyword overlap
    common_keywords = set(result_a['keywords']).intersection(set(result_b['keywords']))
    if common_keywords:
        st.info(f"🔗 Common keywords: {', '.join(common_keywords)}")
    else:
        st.info("🔗 No common keywords found between the texts")

def results_dashboard():
    st.header("Results Dashboard")
    
    if not st.session_state.results:
        st.info("No analysis results available. Please perform some analysis first.")
        return
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(st.session_state.results)
    
    # Dashboard metrics
    st.subheader("Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(df_results))
    
    with col2:
        avg_confidence = df_results['confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    with col3:
        most_common_sentiment = df_results['sentiment'].mode()[0]
        st.metric("Most Common Sentiment", most_common_sentiment)
    
    with col4:
        high_confidence_count = len(df_results[df_results['confidence'] > 0.8])
        st.metric("High Confidence Results", high_confidence_count)
    
    # Visualizations
    st.subheader("Dashboard Visualizations")
    
    # Sentiment over time
    if 'timestamp' in df_results.columns:
        df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
        df_results = df_results.sort_values('timestamp')
        
        fig_timeline = px.scatter(
            df_results,
            x='timestamp',
            y='confidence',
            color='sentiment',
            title="Sentiment Analysis Over Time",
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEGATIVE': '#e74c3c',
                'NEUTRAL': '#f39c12'
            }
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Detailed results management
    st.subheader("Manage Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Results"):
            st.session_state.results = []
            st.rerun()
    
    with col2:
        if st.button("Export All Results"):
            export_utils = ExportUtils()
            json_data = export_utils.export_to_json(st.session_state.results)
            st.download_button(
                label="Download All Results (JSON)",
                data=json_data,
                file_name=f"all_sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Results table
    st.subheader("All Results")
    display_columns = ['timestamp', 'sentiment', 'confidence']
    if 'keywords' in df_results.columns:
        display_columns.append('keywords')
    
    st.dataframe(
        df_results[display_columns],
        use_container_width=True,
        hide_index=True
    )

def highlight_keywords(text, keywords):
    """Highlight keywords in text"""
    if not keywords:
        return text
    
    highlighted = text
    for keyword in keywords:
        highlighted = highlighted.replace(
            keyword, 
            f'<mark style="background-color: #ffeb3b; padding: 1px 2px;">{keyword}</mark>'
        )
    return highlighted

if __name__ == "__main__":
    main()
