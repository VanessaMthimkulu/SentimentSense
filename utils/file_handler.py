import pandas as pd
import io
import csv
import json
import streamlit as st
from typing import Union, List, Dict

class FileHandler:
    def __init__(self):
        self.supported_formats = {
            'text': ['.txt'],
            'csv': ['.csv'],
            'json': ['.json']
        }
    
    def read_text_file(self, uploaded_file) -> str:
        """
        Read content from a text file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: File content
        """
        try:
            # Read file content
            content = uploaded_file.read()
            
            # Decode if bytes
            if isinstance(content, bytes):
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        content = content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use utf-8 with errors ignored
                    content = content.decode('utf-8', errors='ignore')
            
            return content
            
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def read_csv_file(self, uploaded_file) -> pd.DataFrame:
        """
        Read CSV file and return DataFrame
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pd.DataFrame: CSV data
        """
        try:
            # Try different separators
            separators = [',', ';', '\t', '|']
            
            for separator in separators:
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Read CSV
                    df = pd.read_csv(uploaded_file, separator=separator, encoding='utf-8')
                    
                    # Check if DataFrame has meaningful data
                    if len(df.columns) > 1 or len(df) > 0:
                        return df
                        
                except Exception:
                    continue
            
            # If all separators fail, try with default settings
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            return df
            
        except Exception as e:
            # Try with different encoding
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
                return df
            except Exception:
                raise Exception(f"Error reading CSV file: {str(e)}")
    
    def read_json_file(self, uploaded_file) -> Union[Dict, List]:
        """
        Read JSON file and return data
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Union[Dict, List]: JSON data
        """
        try:
            content = uploaded_file.read()
            
            # Decode if bytes
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # Parse JSON
            data = json.loads(content)
            return data
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading JSON file: {str(e)}")
    
    def validate_file_format(self, uploaded_file) -> bool:
        """
        Validate if uploaded file format is supported
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            bool: True if format is supported
        """
        file_extension = f".{uploaded_file.name.split('.')[-1].lower()}"
        
        for format_type, extensions in self.supported_formats.items():
            if file_extension in extensions:
                return True
        
        return False
    
    def get_file_info(self, uploaded_file) -> Dict:
        """
        Get information about uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dict: File information
        """
        try:
            file_info = {
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': uploaded_file.type,
                'size_mb': round(uploaded_file.size / (1024 * 1024), 2)
            }
            return file_info
        except Exception as e:
            return {'error': str(e)}
    
    def process_batch_csv(self, uploaded_file, text_column: str, id_column: str = None) -> List[Dict]:
        """
        Process CSV file for batch analysis
        
        Args:
            uploaded_file: Streamlit uploaded file object
            text_column (str): Name of column containing text
            id_column (str): Name of column containing IDs (optional)
            
        Returns:
            List[Dict]: List of processed records
        """
        try:
            df = self.read_csv_file(uploaded_file)
            
            # Validate columns exist
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV")
            
            if id_column and id_column not in df.columns:
                raise ValueError(f"ID column '{id_column}' not found in CSV")
            
            # Process records
            records = []
            for index, row in df.iterrows():
                record = {
                    'text': str(row[text_column]),
                    'id': row[id_column] if id_column else index,
                    'row_number': index + 1
                }
                records.append(record)
            
            return records
            
        except Exception as e:
            raise Exception(f"Error processing batch CSV: {str(e)}")
    
    def save_results_to_csv(self, results: List[Dict], filename: str = None) -> str:
        """
        Convert results to CSV format
        
        Args:
            results (List[Dict]): Analysis results
            filename (str): Optional filename
            
        Returns:
            str: CSV content as string
        """
        try:
            if not results:
                return ""
            
            # Create DataFrame from results
            df = pd.DataFrame(results)
            
            # Convert to CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_content = csv_buffer.getvalue()
            
            return csv_content
            
        except Exception as e:
            raise Exception(f"Error converting to CSV: {str(e)}")
    
    def save_results_to_json(self, results: List[Dict], filename: str = None) -> str:
        """
        Convert results to JSON format
        
        Args:
            results (List[Dict]): Analysis results
            filename (str): Optional filename
            
        Returns:
            str: JSON content as string
        """
        try:
            if not results:
                return "{}"
            
            # Create JSON with metadata
            json_data = {
                'metadata': {
                    'total_results': len(results),
                    'generated_at': pd.Timestamp.now().isoformat(),
                    'version': '1.0'
                },
                'results': results
            }
            
            # Convert to JSON string
            json_content = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            return json_content
            
        except Exception as e:
            raise Exception(f"Error converting to JSON: {str(e)}")
    
    def create_sample_csv(self) -> str:
        """
        Create a sample CSV for users to understand the format
        
        Returns:
            str: Sample CSV content
        """
        sample_data = {
            'id': [1, 2, 3, 4, 5],
            'text': [
                "This product is amazing! I love it.",
                "The service was terrible, very disappointed.",
                "It's okay, nothing special but works fine.",
                "Excellent quality and fast delivery. Highly recommended!",
                "Worst purchase ever. Complete waste of money."
            ],
            'category': ['Product Review', 'Service Review', 'Product Review', 'Product Review', 'Product Review']
        }
        
        df = pd.DataFrame(sample_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    
    def validate_csv_structure(self, uploaded_file) -> Dict:
        """
        Validate CSV structure and provide suggestions
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Dict: Validation results and suggestions
        """
        try:
            df = self.read_csv_file(uploaded_file)
            
            validation_result = {
                'valid': True,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'text_columns': [],
                'id_columns': [],
                'suggestions': []
            }
            
            # Find potential text columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains text (average length > 10)
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 10:
                        validation_result['text_columns'].append(col)
                    else:
                        validation_result['id_columns'].append(col)
            
            # Provide suggestions
            if not validation_result['text_columns']:
                validation_result['suggestions'].append("No suitable text columns found. Ensure you have columns with text content.")
            
            if len(validation_result['text_columns']) > 1:
                validation_result['suggestions'].append("Multiple text columns found. You'll need to select which one to analyze.")
            
            if validation_result['row_count'] == 0:
                validation_result['valid'] = False
                validation_result['suggestions'].append("CSV file is empty.")
            
            if validation_result['row_count'] > 1000:
                validation_result['suggestions'].append(f"Large file detected ({validation_result['row_count']} rows). Processing may take some time.")
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'suggestions': ['Please check that your file is a valid CSV format.']
            }
