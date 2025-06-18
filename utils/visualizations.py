import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_sentiment_chart(data):
    """
    Create a sentiment distribution chart
    
    Args:
        data (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        plotly.graph_objects.Figure: Sentiment chart
    """
    if isinstance(data, list):
        # Convert list to DataFrame
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Count sentiments
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
    else:
        sentiment_counts = df.groupby('Sentiment').size()
    
    # Color mapping
    color_map = {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c', 
        'NEUTRAL': '#f39c12'
    }
    
    # Create pie chart
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=color_map
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_confidence_chart(confidence, sentiment):
    """
    Create a confidence gauge chart
    
    Args:
        confidence (float): Confidence score
        sentiment (str): Sentiment label
        
    Returns:
        plotly.graph_objects.Figure: Confidence gauge chart
    """
    # Color based on sentiment
    color_map = {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c',
        'NEUTRAL': '#f39c12'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence Score<br><span style='font-size:0.8em;color:gray'>{sentiment}</span>"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color_map.get(sentiment, '#3498db')},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_comparison_chart(comparison_data):
    """
    Create a comparison chart for multiple texts
    
    Args:
        comparison_data (pd.DataFrame): DataFrame with comparison data
        
    Returns:
        plotly.graph_objects.Figure: Comparison chart
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sentiment Comparison', 'Confidence Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Color mapping
    color_map = {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c',
        'NEUTRAL': '#f39c12'
    }
    
    # Sentiment comparison
    colors = [color_map.get(sentiment, '#3498db') for sentiment in comparison_data['Sentiment']]
    
    fig.add_trace(
        go.Bar(
            x=comparison_data['Source'],
            y=[1] * len(comparison_data),  # All bars same height for sentiment
            marker_color=colors,
            text=comparison_data['Sentiment'],
            textposition='inside',
            name='Sentiment',
            hovertemplate='<b>%{x}</b><br>Sentiment: %{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Confidence comparison
    fig.add_trace(
        go.Bar(
            x=comparison_data['Source'],
            y=comparison_data['Confidence'],
            marker_color=colors,
            text=[f"{conf:.2%}" for conf in comparison_data['Confidence']],
            textposition='outside',
            name='Confidence',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Sentiment Analysis Comparison",
        showlegend=False,
        height=400
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Sentiment", row=1, col=1, showticklabels=False)
    fig.update_yaxes(title_text="Confidence Score", row=1, col=2, range=[0, 1])
    
    return fig

def create_batch_timeline(batch_data):
    """
    Create timeline visualization for batch processing results
    
    Args:
        batch_data (pd.DataFrame): DataFrame with batch results
        
    Returns:
        plotly.graph_objects.Figure: Timeline chart
    """
    if 'timestamp' not in batch_data.columns:
        return None
    
    # Convert timestamp to datetime if it's not already
    batch_data['timestamp'] = pd.to_datetime(batch_data['timestamp'])
    
    # Color mapping
    color_map = {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c',
        'NEUTRAL': '#f39c12'
    }
    
    fig = px.scatter(
        batch_data,
        x='timestamp',
        y='confidence',
        color='sentiment',
        size_max=10,
        title="Sentiment Analysis Timeline",
        labels={
            'timestamp': 'Analysis Time',
            'confidence': 'Confidence Score',
            'sentiment': 'Sentiment'
        },
        color_discrete_map=color_map,
        hover_data=['sentiment', 'confidence']
    )
    
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color='white')),
        hovertemplate='<b>%{customdata[0]}</b><br>Time: %{x}<br>Confidence: %{y:.2%}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Analysis Time",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_confidence_distribution(confidence_scores):
    """
    Create confidence score distribution histogram
    
    Args:
        confidence_scores (list): List of confidence scores
        
    Returns:
        plotly.graph_objects.Figure: Histogram chart
    """
    fig = px.histogram(
        x=confidence_scores,
        nbins=20,
        title="Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Frequency'},
        color_discrete_sequence=['#3498db']
    )
    
    # Add vertical lines for thresholds
    fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Low Confidence", annotation_position="top")
    fig.add_vline(x=0.8, line_dash="dash", line_color="green", 
                  annotation_text="High Confidence", annotation_position="top")
    
    fig.update_layout(
        height=400,
        xaxis_title="Confidence Score",
        yaxis_title="Number of Analyses",
        showlegend=False
    )
    
    return fig

def create_keyword_cloud_chart(keywords_data):
    """
    Create a keyword frequency chart (as bar chart since we can't use wordcloud)
    
    Args:
        keywords_data (dict): Dictionary with keyword frequencies
        
    Returns:
        plotly.graph_objects.Figure: Keyword frequency chart
    """
    if not keywords_data:
        return None
    
    # Sort keywords by frequency
    sorted_keywords = sorted(keywords_data.items(), key=lambda x: x[1], reverse=True)
    keywords, frequencies = zip(*sorted_keywords[:15])  # Top 15 keywords
    
    fig = px.bar(
        x=list(frequencies),
        y=list(keywords),
        orientation='h',
        title="Top Keywords",
        labels={'x': 'Frequency', 'y': 'Keywords'},
        color=list(frequencies),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False,
        coloraxis_showscale=False
    )
    
    return fig

def create_sentiment_confidence_scatter(data):
    """
    Create scatter plot of sentiment vs confidence
    
    Args:
        data (pd.DataFrame): DataFrame with sentiment and confidence data
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot
    """
    # Create numerical mapping for sentiment
    sentiment_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    data['sentiment_numeric'] = data['sentiment'].map(sentiment_mapping)
    
    color_map = {
        'POSITIVE': '#2ecc71',
        'NEGATIVE': '#e74c3c',
        'NEUTRAL': '#f39c12'
    }
    
    fig = px.scatter(
        data,
        x='sentiment_numeric',
        y='confidence',
        color='sentiment',
        title="Sentiment vs Confidence Distribution",
        labels={
            'sentiment_numeric': 'Sentiment',
            'confidence': 'Confidence Score'
        },
        color_discrete_map=color_map,
        hover_data=['sentiment', 'confidence']
    )
    
    # Update x-axis to show sentiment labels
    fig.update_xaxes(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=['Negative', 'Neutral', 'Positive']
    )
    
    fig.update_layout(
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig
