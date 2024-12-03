import gradio as gr
import json
import pandas as pd
from typing import Dict, List, Union
import argparse

def load_json(file_path: str) -> List[Dict]:
    """Load and parse JSON review file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except Exception as e:
        raise gr.Error(f"Error loading file: {str(e)}")

def format_list(items) -> str:
    """Format list items with bullet points, handling None and non-list inputs"""
    if not items or not isinstance(items, list):
        return ""
    # Using regular newlines for display
    return '\n'.join(f"• {item}" for item in items if item)

def convert_review_to_df(reviews: List[Dict]) -> pd.DataFrame:
    """Convert review data to DataFrame with formatted display"""
    if not reviews:
        return pd.DataFrame()
    
    # Convert reviews to DataFrame directly
    df = pd.DataFrame(reviews)
    
    # Format the features columns if they exist
    if 'positive_features' in df.columns:
        df['positive_features'] = df['positive_features'].fillna('').apply(format_list)
    if 'negative_features' in df.columns:
        df['negative_features'] = df['negative_features'].fillna('').apply(format_list)
    
    # Reorder and rename columns if they exist
    column_mapping = {
        'product_id': 'Product ID', 
        'score': 'Score', 
        'summarized_review': 'Summary', 
        'raw_review': 'Review Text', 
        'positive_features': 'Positive Features', 
        'negative_features': 'Negative Features'
    }
    
    # Only include columns that exist in the DataFrame
    cols = [col for col in column_mapping.keys() if col in df.columns]
    df = df[cols]
    
    # Rename columns
    df.columns = [column_mapping[col] for col in df.columns]
    
    return df

def parse_feature_list(features: str) -> List[str]:
    """Convert formatted feature list back to Python list"""
    if not features or not isinstance(features, str):
        return []
    return [item.strip('• ').strip() for item in features.split('\n') if item.strip('• ').strip()]

def save_reviews(df: pd.DataFrame, output_path: str) -> str:
    """Save reviews back to JSON format"""
    if df is None or df.empty:
        return "No data to save"
    
    # Reverse the column mapping
    reverse_mapping = {
        'Product ID': 'product_id',
        'Score': 'score',
        'Summary': 'summarized_review',
        'Review Text': 'raw_review',
        'Positive Features': 'positive_features',
        'Negative Features': 'negative_features'
    }
    
    # Rename columns back to original format
    df = df.rename(columns=reverse_mapping)
    
    # Convert DataFrame back to dictionaries
    reviews = df.to_dict('records')
    
    # Convert feature strings back to lists
    for review in reviews:
        if 'positive_features' in review:
            review['positive_features'] = parse_feature_list(review['positive_features'])
        if 'negative_features' in review:
            review['negative_features'] = parse_feature_list(review['negative_features'])
    
    if not output_path.endswith('.json'):
        output_path += '.json'
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
        return f"Reviews successfully saved to {output_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

def create_app(input_path: str, output_path: str):
    with gr.Blocks() as app:
        gr.Markdown("# Product Review Viewer")
        
        # Initialize DataFrame first
        initial_data = convert_review_to_df(load_json(input_path))
        
        with gr.Row():
            # Display component
            data_display = gr.Dataframe(
                value=initial_data,
                interactive=False,
                wrap=True,
                max_height=800,
                row_count=20
            )
            
        with gr.Row():
            save_btn = gr.Button("Save Reviews")
            output_msg = gr.Textbox(label="Status")
        
        # Event handler for save button
        save_btn.click(
            fn=lambda df: save_reviews(df, output_path),
            inputs=[data_display],
            outputs=output_msg
        )
    
    return app

def parse_arguments():
    parser = argparse.ArgumentParser(description='Product Review Viewer')
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input JSON file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                      help='Output JSON file path')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    app = create_app(args.input, args.output)
    app.launch(share=True)
