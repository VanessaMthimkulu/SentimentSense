import json
import csv
import io
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

class ExportUtils:
    def __init__(self):
        self.export_formats = ['csv', 'json', 'txt']
    
    def export_to_csv(self, results: List[Dict]) -> str:
        """
        Export results to CSV format
        
        Args:
            results (List[Dict]): Analysis results
            
        Returns:
            str: CSV content as string
        """
        if not results:
            return ""
        
        try:
            # Flatten nested data if needed
            flattened_results = []
            for result in results:
                flattened_result = self._flatten_dict(result)
                flattened_results.append(flattened_result)
            
            # Create DataFrame
            df = pd.DataFrame(flattened_results)
            
            # Reorder columns for better readability
            column_order = []
            if 'id' in df.columns:
                column_order.append('id')
            if 'timestamp' in df.columns:
                column_order.append('timestamp')
            if 'sentiment' in df.columns:
                column_order.append('sentiment')
            if 'confidence' in df.columns:
                column_order.append('confidence')
            if 'text' in df.columns:
                column_order.append('text')
            
            # Add remaining columns
            remaining_cols = [col for col in df.columns if col not in column_order]
            column_order.extend(remaining_cols)
            
            # Reorder DataFrame
            df = df[column_order]
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            
            return csv_buffer.getvalue()
            
        except Exception as e:
            st.error(f"Error exporting to CSV: {str(e)}")
            return ""
    
    def export_to_json(self, results: List[Dict]) -> str:
        """
        Export results to JSON format
        
        Args:
            results (List[Dict]): Analysis results
            
        Returns:
            str: JSON content as string
        """
        if not results:
            return "{}"
        
        try:
            # Create comprehensive JSON structure
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_results': len(results),
                    'format_version': '1.0',
                    'generated_by': 'Sentiment Analysis Dashboard'
                },
                'summary': self._generate_summary(results),
                'results': results
            }
            
            # Convert to JSON string with proper formatting
            json_content = json.dumps(
                export_data, 
                indent=2, 
                ensure_ascii=False,
                default=self._json_serializer
            )
            
            return json_content
            
        except Exception as e:
            st.error(f"Error exporting to JSON: {str(e)}")
            return "{}"
    
    def export_to_txt(self, results: List[Dict]) -> str:
        """
        Export results to readable text format
        
        Args:
            results (List[Dict]): Analysis results
            
        Returns:
            str: Text content as string
        """
        if not results:
            return ""
        
        try:
            lines = []
            lines.append("SENTIMENT ANALYSIS RESULTS")
            lines.append("=" * 50)
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Total Results: {len(results)}")
            lines.append("")
            
            # Add summary
            summary = self._generate_summary(results)
            lines.append("SUMMARY")
            lines.append("-" * 20)
            for key, value in summary.items():
                lines.append(f"{key}: {value}")
            lines.append("")
            
            # Add detailed results
            lines.append("DETAILED RESULTS")
            lines.append("-" * 30)
            
            for i, result in enumerate(results, 1):
                lines.append(f"\n{i}. Analysis Result")
                lines.append(f"   ID: {result.get('id', 'N/A')}")
                lines.append(f"   Sentiment: {result.get('sentiment', 'N/A')}")
                lines.append(f"   Confidence: {result.get('confidence', 0):.2%}")
                
                if 'keywords' in result and result['keywords']:
                    lines.append(f"   Keywords: {', '.join(result['keywords'])}")
                
                if 'text' in result:
                    text = result['text']
                    if len(text) > 200:
                        text = text[:200] + "..."
                    lines.append(f"   Text: {text}")
                
                if 'timestamp' in result:
                    lines.append(f"   Analyzed: {result['timestamp']}")
                
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            st.error(f"Error exporting to text: {str(e)}")
            return ""
    
    def create_export_report(self, results: List[Dict]) -> Dict[str, str]:
        """
        Create export report with multiple formats
        
        Args:
            results (List[Dict]): Analysis results
            
        Returns:
            Dict[str, str]: Dictionary with different format exports
        """
        report = {}
        
        try:
            report['csv'] = self.export_to_csv(results)
            report['json'] = self.export_to_json(results)
            report['txt'] = self.export_to_txt(results)
            
            return report
            
        except Exception as e:
            st.error(f"Error creating export report: {str(e)}")
            return {}
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        Flatten nested dictionary
        
        Args:
            d (Dict): Dictionary to flatten
            parent_key (str): Parent key for nested keys
            sep (str): Separator for nested keys
            
        Returns:
            Dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                if v and isinstance(v[0], (str, int, float)):
                    items.append((new_key, ', '.join(map(str, v))))
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics from results
        
        Args:
            results (List[Dict]): Analysis results
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not results:
            return {}
        
        try:
            df = pd.DataFrame(results)
            
            summary = {
                'total_analyses': len(results),
                'average_confidence': f"{df['confidence'].mean():.2%}" if 'confidence' in df.columns else 'N/A'
            }
            
            # Sentiment distribution
            if 'sentiment' in df.columns:
                sentiment_counts = df['sentiment'].value_counts()
                for sentiment, count in sentiment_counts.items():
                    summary[f'{sentiment.lower()}_count'] = count
                    summary[f'{sentiment.lower()}_percentage'] = f"{(count/len(results)*100):.1f}%"
            
            # Confidence levels
            if 'confidence' in df.columns:
                high_conf = len(df[df['confidence'] > 0.8])
                medium_conf = len(df[(df['confidence'] > 0.5) & (df['confidence'] <= 0.8)])
                low_conf = len(df[df['confidence'] <= 0.5])
                
                summary['high_confidence_count'] = high_conf
                summary['medium_confidence_count'] = medium_conf
                summary['low_confidence_count'] = low_conf
            
            # Time range
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
                summary['analysis_period_start'] = timestamps.min().strftime('%Y-%m-%d %H:%M:%S')
                summary['analysis_period_end'] = timestamps.max().strftime('%Y-%m-%d %H:%M:%S')
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def _json_serializer(self, obj):
        """
        JSON serializer for non-serializable objects
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serialized object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def get_export_filename(self, format_type: str, prefix: str = "sentiment_analysis") -> str:
        """
        Generate filename for export
        
        Args:
            format_type (str): Export format (csv, json, txt)
            prefix (str): Filename prefix
            
        Returns:
            str: Generated filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}.{format_type}"
    
    def validate_export_data(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Validate data before export
        
        Args:
            results (List[Dict]): Data to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        try:
            if not results:
                validation['valid'] = False
                validation['errors'].append("No data to export")
                return validation
            
            # Check data structure
            required_fields = ['sentiment', 'confidence']
            missing_fields = []
            
            for field in required_fields:
                if not any(field in result for result in results):
                    missing_fields.append(field)
            
            if missing_fields:
                validation['warnings'].append(f"Missing fields: {', '.join(missing_fields)}")
            
            # Check data quality
            if 'confidence' in results[0]:
                confidences = [r.get('confidence', 0) for r in results]
                low_confidence_count = sum(1 for c in confidences if c < 0.5)
                
                if low_confidence_count > len(results) * 0.5:
                    validation['warnings'].append(f"High number of low confidence results: {low_confidence_count}")
            
            # Generate stats
            validation['stats'] = {
                'total_results': len(results),
                'estimated_csv_size': len(str(results)) * 1.2,  # Rough estimate
                'estimated_json_size': len(json.dumps(results)) * 1.1
            }
            
            return validation
            
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(str(e))
            return validation
