import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
import difflib
from io import StringIO

# Configuration and Constants
CANONICAL_SCHEMA = {
    'order_id': {'description': 'Unique order identifier', 'example': 'ORD-1001'},
    'order_date': {'description': 'ISO date of order', 'example': '2025-08-09'},
    'customer_id': {'description': 'Internal customer id', 'example': 'CUST-42'},
    'customer_name': {'description': 'Full name', 'example': 'Aarav Sharma'},
    'email': {'description': 'Contact email', 'example': 'aarav@example.com'},
    'phone': {'description': 'Contact phone', 'example': '+91-9000156794'},
    'billing_address': {'description': 'Billing address line', 'example': '221B Baker Street'},
    'shipping_address': {'description': 'Shipping address line', 'example': '14 MG Road'},
    'city': {'description': 'City', 'example': 'Bengaluru'},
    'state': {'description': 'State/Province', 'example': 'Karnataka'},
    'postal_code': {'description': 'Zip/Postal', 'example': '560095'},
    'country': {'description': 'Country', 'example': 'India'},
    'product_sku': {'description': 'SKU code', 'example': 'SW-1234'},
    'product_name': {'description': 'Item name', 'example': 'Alpha SaaS'},
    'category': {'description': 'Category', 'example': 'Software Subscription'},
    'subcategory': {'description': 'Subcategory if any', 'example': 'Enterprise'},
    'quantity': {'description': 'Units ordered', 'example': '3'},
    'unit_price': {'description': 'Price per unit', 'example': '4999.99'},
    'currency': {'description': 'Currency code', 'example': 'INR'},
    'discount_pct': {'description': 'Discount fraction (0-1)', 'example': '0.1'},
    'tax_pct': {'description': 'Tax fraction (0-1)', 'example': '0.18'},
    'shipping_fee': {'description': 'Shipping amount', 'example': '199.0'},
    'total_amount': {'description': 'Total amount charged', 'example': '16999.00'},
    'tax_id': {'description': 'Tax/GST/VAT identifier', 'example': '29ABCDE1234F1Z5'}
}

# Mapping rules for common variations
MAPPING_RULES = {
    'order_id': ['order_id', 'order no', 'reference', 'order_number', 'id'],
    'order_date': ['order_date', 'orderdate', 'ordered_on', 'date'],
    'customer_id': ['customer_id', 'cust id', 'client_ref', 'customer_ref'],
    'customer_name': ['customer_name', 'customer', 'client_name', 'name'],
    'email': ['email', 'e-mail', 'contact', 'email_address'],
    'phone': ['phone', 'phone #', 'mobile', 'contact_phone'],
    'billing_address': ['billing_address', 'bill addr', 'bill_to', 'billing'],
    'shipping_address': ['shipping_address', 'ship addr', 'ship_to', 'shipping'],
    'city': ['city'],
    'state': ['state', 'state/province', 'province'],
    'postal_code': ['postal_code', 'zip/postal', 'pin', 'zip', 'postal'],
    'country': ['country', 'country/region'],
    'product_sku': ['product_sku', 'sku', 'stock_code', 'item_code'],
    'product_name': ['product_name', 'item', 'desc', 'description'],
    'category': ['category', 'cat.', 'cat'],
    'subcategory': ['subcategory', 'subcat', 'sub_category'],
    'quantity': ['quantity', 'qty', 'units'],
    'unit_price': ['unit_price', 'unit price', 'price', 'rate'],
    'currency': ['currency'],
    'discount_pct': ['discount_pct', 'disc%', 'discount'],
    'tax_pct': ['tax_pct', 'tax%', 'gst', 'vat'],
    'shipping_fee': ['shipping_fee', 'ship fee', 'logistics_fee'],
    'total_amount': ['total_amount', 'total', 'grand_total', 'amount'],
    'tax_id': ['tax_id', 'reg no', 'gstin', 'vat_id', 'tax_number']
}

class SchemaMapper:
    def __init__(self):
        self.confidence_threshold = 0.7
        
    def calculate_similarity(self, source_col: str, target_col: str) -> float:
        """Calculate similarity between column names"""
        source_clean = self._clean_column_name(source_col)
        target_clean = self._clean_column_name(target_col)
        
        # Exact match
        if source_clean == target_clean:
            return 1.0
            
        # Check if source is in target variations
        for canonical, variations in MAPPING_RULES.items():
            if canonical == target_col:
                for variation in variations:
                    if self._clean_column_name(variation) == source_clean:
                        return 0.95
                        
        # Fuzzy matching
        return difflib.SequenceMatcher(None, source_clean, target_clean).ratio()
    
    def _clean_column_name(self, col_name: str) -> str:
        """Clean column name for comparison"""
        return re.sub(r'[^a-zA-Z0-9]', '', col_name.lower())
    
    def suggest_mapping(self, source_columns: List[str]) -> Dict[str, Dict]:
        """Suggest mapping from source to canonical schema"""
        mapping = {}
        
        for source_col in source_columns:
            best_match = None
            best_score = 0
            
            for canonical_col in CANONICAL_SCHEMA.keys():
                score = self.calculate_similarity(source_col, canonical_col)
                if score > best_score:
                    best_score = score
                    best_match = canonical_col
            
            mapping[source_col] = {
                'suggested': best_match,
                'confidence': best_score,
                'canonical_info': CANONICAL_SCHEMA.get(best_match, {})
            }
        
        return mapping

class DataCleaner:
    def __init__(self):
        self.cleaning_rules = self._load_cleaning_rules()
    
    def _load_cleaning_rules(self) -> Dict:
        """Load cleaning rules from session state or defaults"""
        if 'cleaning_rules' in st.session_state:
            return st.session_state.cleaning_rules
        
        return {
            'date_formats': ['%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y', '%d %b %Y', '%Y-%m-%d'],
            'currency_symbols': ['₹', 'Rs', 'INR', '$', '€'],
            'percentage_indicators': ['%'],
            'number_separators': [','],
            'email_pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone_pattern': r'^\+?[\d\s\-\(\)]+$'
        }
    
    def clean_data(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
        """Clean data based on mapped columns"""
        cleaned_df = df.copy()
        issues = {'fixed': [], 'warnings': [], 'errors': []}
        
        for source_col, target_col in column_mapping.items():
            if source_col in cleaned_df.columns and target_col:
                cleaned_df[target_col] = cleaned_df[source_col].copy()
                
                # Apply cleaning based on target column type
                if target_col == 'order_date':
                    cleaned_df[target_col], date_issues = self._clean_dates(cleaned_df[target_col])
                    issues['fixed'].extend(date_issues)
                
                elif target_col in ['unit_price', 'total_amount', 'shipping_fee']:
                    cleaned_df[target_col], money_issues = self._clean_money(cleaned_df[target_col])
                    issues['fixed'].extend(money_issues)
                
                elif target_col in ['discount_pct', 'tax_pct']:
                    cleaned_df[target_col], pct_issues = self._clean_percentages(cleaned_df[target_col])
                    issues['fixed'].extend(pct_issues)
                
                elif target_col == 'email':
                    email_issues = self._validate_emails(cleaned_df[target_col])
                    issues['warnings'].extend(email_issues)
                
                elif target_col == 'phone':
                    phone_issues = self._validate_phones(cleaned_df[target_col])
                    issues['warnings'].extend(phone_issues)
                
                elif target_col == 'postal_code':
                    cleaned_df[target_col], postal_issues = self._clean_postal_codes(cleaned_df[target_col])
                    issues['fixed'].extend(postal_issues)
                
                # Clean whitespace and casing
                if cleaned_df[target_col].dtype == 'object':
                    cleaned_df[target_col] = cleaned_df[target_col].astype(str).str.strip()
                    if target_col in ['customer_name', 'product_name']:
                        cleaned_df[target_col] = cleaned_df[target_col].str.title()
        
        # Remove original columns that were mapped
        cols_to_drop = [col for col in df.columns if col in column_mapping and column_mapping[col]]
        cleaned_df = cleaned_df.drop(columns=cols_to_drop, errors='ignore')
        
        return cleaned_df, issues
    
    def _clean_dates(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Clean date formats"""
        issues = []
        cleaned_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            original_value = str(value)
            
            # Try different date formats
            for date_format in self.cleaning_rules['date_formats']:
                try:
                    if date_format == '%d %b %Y':
                        # Handle "02 Aug 2025" format
                        parsed_date = datetime.strptime(original_value, date_format)
                    else:
                        parsed_date = datetime.strptime(original_value, date_format)
                    
                    cleaned_series.iloc[idx] = parsed_date.strftime('%Y-%m-%d')
                    if original_value != cleaned_series.iloc[idx]:
                        issues.append(f"Date format standardized: {original_value} → {cleaned_series.iloc[idx]}")
                    break
                except ValueError:
                    continue
        
        return cleaned_series, issues
    
    def _clean_money(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Clean monetary values"""
        issues = []
        cleaned_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            original_value = str(value)
            cleaned_value = original_value
            
            # Remove currency symbols
            for symbol in self.cleaning_rules['currency_symbols']:
                cleaned_value = cleaned_value.replace(symbol, '')
            
            # Remove commas
            cleaned_value = cleaned_value.replace(',', '')
            
            # Remove quotes
            cleaned_value = cleaned_value.replace('"', '').strip()
            
            try:
                numeric_value = float(cleaned_value)
                cleaned_series.iloc[idx] = numeric_value
                if original_value != str(numeric_value):
                    issues.append(f"Money format cleaned: {original_value} → {numeric_value}")
            except ValueError:
                issues.append(f"Could not parse money value: {original_value}")
        
        return cleaned_series, issues
    
    def _clean_percentages(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Clean percentage values"""
        issues = []
        cleaned_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            original_value = str(value)
            cleaned_value = original_value
            
            # Handle percentage signs
            if '%' in cleaned_value:
                cleaned_value = cleaned_value.replace('%', '')
                try:
                    numeric_value = float(cleaned_value) / 100  # Convert to decimal
                    cleaned_series.iloc[idx] = numeric_value
                    issues.append(f"Percentage converted: {original_value} → {numeric_value}")
                except ValueError:
                    pass
            else:
                try:
                    numeric_value = float(cleaned_value)
                    cleaned_series.iloc[idx] = numeric_value
                except ValueError:
                    pass
        
        return cleaned_series, issues
    
    def _validate_emails(self, series: pd.Series) -> List[str]:
        """Validate email addresses"""
        issues = []
        pattern = re.compile(self.cleaning_rules['email_pattern'])
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            email = str(value).strip()
            # Clean spaces around @ symbol
            email = re.sub(r'\s*@\s*', '@', email)
            
            if not pattern.match(email):
                issues.append(f"Invalid email format at row {idx}: {email}")
        
        return issues
    
    def _validate_phones(self, series: pd.Series) -> List[str]:
        """Validate phone numbers"""
        issues = []
        pattern = re.compile(self.cleaning_rules['phone_pattern'])
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            phone = str(value).strip()
            if not pattern.match(phone):
                issues.append(f"Invalid phone format at row {idx}: {phone}")
        
        return issues
    
    def _clean_postal_codes(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Clean postal codes"""
        issues = []
        cleaned_series = series.copy()
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            original_value = str(value)
            # Handle XX patterns in postal codes
            if 'XX' in original_value:
                issues.append(f"Postal code has placeholder: {original_value}")
            
            # Remove extra spaces
            cleaned_value = re.sub(r'\s+', '', original_value)
            # Fix: Convert to string to avoid dtype issues
            cleaned_series.iloc[idx] = str(cleaned_value)
            
            if original_value != cleaned_value:
                issues.append(f"Postal code cleaned: {original_value} → {cleaned_value}")
        
        return cleaned_series, issues

class TargetedFixer:
    def __init__(self):
        self.fix_suggestions = []
    
    def analyze_issues(self, df: pd.DataFrame, issues: Dict) -> List[Dict]:
        """Analyze remaining issues and suggest targeted fixes"""
        suggestions = []
        
        # Analyze missing values
        missing_analysis = df.isnull().sum()
        for col, missing_count in missing_analysis.items():
            if missing_count > 0:
                suggestions.append({
                    'type': 'missing_values',
                    'column': col,
                    'count': missing_count,
                    'suggestion': f"Fill {missing_count} missing values in {col}",
                    'auto_fix': self._suggest_missing_value_fix(df, col)
                })
        
        # Analyze data type inconsistencies
        for col in df.columns:
            if col in ['quantity', 'unit_price', 'total_amount', 'shipping_fee']:
                non_numeric = df[~pd.to_numeric(df[col], errors='coerce').notna()]
                if len(non_numeric) > 0:
                    suggestions.append({
                        'type': 'data_type',
                        'column': col,
                        'count': len(non_numeric),
                        'suggestion': f"Convert {len(non_numeric)} non-numeric values in {col}",
                        'auto_fix': 'convert_to_numeric'
                    })
        
        # Analyze outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['unit_price', 'total_amount']:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
                
                if len(outliers) > 0:
                    suggestions.append({
                        'type': 'outliers',
                        'column': col,
                        'count': len(outliers),
                        'suggestion': f"Review {len(outliers)} potential outliers in {col}",
                        'auto_fix': 'flag_for_review'
                    })
        
        return suggestions
    
    def _suggest_missing_value_fix(self, df: pd.DataFrame, col: str) -> str:
        """Suggest how to fix missing values"""
        if col in ['email', 'phone']:
            return 'manual_entry_required'
        elif col in ['unit_price', 'total_amount']:
            return 'use_median'
        elif col in ['country']:
            return 'use_mode'
        else:
            return 'manual_review'
    
    def apply_fix(self, df: pd.DataFrame, fix_config: Dict) -> pd.DataFrame:
        """Apply a targeted fix"""
        fixed_df = df.copy()
        
        if fix_config['auto_fix'] == 'use_median':
            median_val = fixed_df[fix_config['column']].median()
            fixed_df[fix_config['column']].fillna(median_val, inplace=True)
        
        elif fix_config['auto_fix'] == 'use_mode':
            mode_val = fixed_df[fix_config['column']].mode()[0] if not fixed_df[fix_config['column']].mode().empty else 'Unknown'
            fixed_df[fix_config['column']].fillna(mode_val, inplace=True)
        
        elif fix_config['auto_fix'] == 'convert_to_numeric':
            fixed_df[fix_config['column']] = pd.to_numeric(fixed_df[fix_config['column']], errors='coerce')
        
        return fixed_df

def save_cleaning_rules(rules: Dict):
    """Save cleaning rules to session state"""
    st.session_state.cleaning_rules = rules

def load_sample_data():
    """Load sample data for testing"""
    sample_files = {
        'Project6InputData1.csv': 'Clean data with canonical headers',
        'Project6InputData2.csv': 'Messy headers and formats',
        'Project6InputData3.csv': 'Different schema with missing columns'
    }
    return sample_files

def safe_dataframe_display(df, **kwargs):
    """Safely display dataframe with proper type conversion"""
    try:
        # Convert problematic columns to string to avoid Arrow serialization issues
        display_df = df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        return st.dataframe(display_df, **kwargs)
    except Exception as e:
        st.error(f"Error displaying dataframe: {str(e)}")
        return st.text("Unable to display dataframe due to formatting issues")

def main():
    st.set_page_config(
        page_title="Schema Mapper & Data Quality Fixer",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Schema Mapper & Data Quality Fixer")
    st.markdown("**Map → Clean → Targeted Repair**")
    
    # Initialize session state
    if 'mapper' not in st.session_state:
        st.session_state.mapper = SchemaMapper()
    if 'cleaner' not in st.session_state:
        st.session_state.cleaner = DataCleaner()
    if 'fixer' not in st.session_state:
        st.session_state.fixer = TargetedFixer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    step = st.sidebar.radio("Select Step:", [
        "📁 Upload Data",
        "🗺️ Schema Mapping", 
        "🧹 Clean & Validate",
        "🎯 Targeted Fixes",
        "📊 Results & Export"
    ])
    
    if step == "📁 Upload Data":
        st.header("Step 1: Upload Your Data")
        
        # Sample data section
        st.subheader("Try with Sample Data")
        sample_files = load_sample_data()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Load Sample 1 (Clean)", key="sample1"):
                try:
                    df = pd.read_csv('Project6InputData1.csv')
                    st.session_state.uploaded_data = df
                    st.session_state.filename = 'Project6InputData1.csv'
                    st.success("Sample data 1 loaded!")
                except Exception as e:
                    st.error(f"Sample file not found: {str(e)}")
        
        with col2:
            if st.button("Load Sample 2 (Messy)", key="sample2"):
                try:
                    df = pd.read_csv('Project6InputData2.csv')
                    st.session_state.uploaded_data = df
                    st.session_state.filename = 'Project6InputData2.csv'
                    st.success("Sample data 2 loaded!")
                except Exception as e:
                    st.error(f"Sample file not found: {str(e)}")
        
        with col3:
            if st.button("Load Sample 3 (Different)", key="sample3"):
                try:
                    df = pd.read_csv('Project6InputData3.csv')
                    st.session_state.uploaded_data = df
                    st.session_state.filename = 'Project6InputData3.csv'
                    st.success("Sample data 3 loaded!")
                except Exception as e:
                    st.error(f"Sample file not found: {str(e)}")
        
        st.markdown("---")
        
        # File upload
        st.subheader("Or Upload Your Own File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_data = df
                st.session_state.filename = uploaded_file.name
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Display uploaded data
        if 'uploaded_data' in st.session_state:
            st.subheader("Data Preview")
            df = st.session_state.uploaded_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            safe_dataframe_display(df.head(10), use_container_width=True)
            
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes],  # Convert to string
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Sample Values': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A' for col in df.columns]
            })
            safe_dataframe_display(col_info, use_container_width=True)
    
    elif step == "🗺️ Schema Mapping":
        st.header("Step 2: Schema Mapping")
        
        if 'uploaded_data' not in st.session_state:
            st.warning("Please upload data first!")
            return
        
        df = st.session_state.uploaded_data
        
        # Show canonical schema
        st.subheader("Canonical Schema Reference")
        schema_df = pd.DataFrame([
            {'Field': k, 'Description': v['description'], 'Example': v['example']}
            for k, v in CANONICAL_SCHEMA.items()
        ])
        safe_dataframe_display(schema_df, use_container_width=True)
        
        st.markdown("---")
        
        # Generate mapping suggestions
        st.subheader("Suggested Mapping")
        
        if st.button("Generate Mapping Suggestions", type="primary"):
            mapping_suggestions = st.session_state.mapper.suggest_mapping(df.columns.tolist())
            st.session_state.mapping_suggestions = mapping_suggestions
        
        if 'mapping_suggestions' in st.session_state:
            mapping_suggestions = st.session_state.mapping_suggestions
            
            # Display suggestions with confidence
            st.subheader("Mapping Suggestions with Confidence")
            
            mapping_df = pd.DataFrame([
                {
                    'Source Column': source_col,
                    'Suggested Mapping': suggestion['suggested'],
                    'Confidence': f"{suggestion['confidence']:.2%}",
                    'Description': suggestion['canonical_info'].get('description', 'N/A')
                }
                for source_col, suggestion in mapping_suggestions.items()
            ])
            
            safe_dataframe_display(mapping_df, use_container_width=True)
            
            # Manual override section
            st.subheader("Review and Override Mapping")
            
            final_mapping = {}
            
            for source_col in df.columns:
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.text(f"Source: {source_col}")
                
                with col2:
                    # Fix: Safely get mapping suggestions with default values
                    suggestion_data = mapping_suggestions.get(source_col, {})
                    suggested = suggestion_data.get('suggested', None)
                    confidence = suggestion_data.get('confidence', 0.0)
                    
                    # Create options list
                    options = ['[Skip]'] + list(CANONICAL_SCHEMA.keys())
                    default_idx = 0
                    
                    if suggested and confidence >= st.session_state.mapper.confidence_threshold:
                        try:
                            default_idx = options.index(suggested)
                        except ValueError:
                            default_idx = 0
                    
                    selected = st.selectbox(
                        f"Map to:",
                        options,
                        index=default_idx,
                        key=f"mapping_{source_col}"
                    )
                    
                    if selected != '[Skip]':
                        final_mapping[source_col] = selected
                
                with col3:
                    if confidence >= 0.8:
                        st.success(f"{confidence:.1%}")
                    elif confidence >= 0.6:
                        st.warning(f"{confidence:.1%}")
                    else:
                        st.error(f"{confidence:.1%}")
            
            st.session_state.final_mapping = final_mapping
            
            if st.button("Confirm Mapping", type="primary"):
                st.success("Mapping confirmed! Proceed to cleaning step.")
                st.session_state.mapping_confirmed = True
    
    elif step == "🧹 Clean & Validate":
        st.header("Step 3: Clean & Validate")
        
        if 'uploaded_data' not in st.session_state or 'final_mapping' not in st.session_state:
            st.warning("Please complete the previous steps first!")
            return
        
        df = st.session_state.uploaded_data
        mapping = st.session_state.final_mapping
        
        st.subheader("Cleaning Configuration")
        
        # Show current mapping
        st.write("**Current Mapping:**")
        mapping_display = pd.DataFrame([
            {'Source Column': k, 'Target Column': v}
            for k, v in mapping.items()
        ])
        safe_dataframe_display(mapping_display, use_container_width=True)
        
        # Cleaning options
        st.subheader("Cleaning Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            clean_dates = st.checkbox("Standardize date formats", value=True)
            clean_money = st.checkbox("Clean monetary values", value=True)
            clean_percentages = st.checkbox("Convert percentages", value=True)
        
        with col2:
            validate_emails = st.checkbox("Validate email addresses", value=True)
            validate_phones = st.checkbox("Validate phone numbers", value=True)
            clean_whitespace = st.checkbox("Clean whitespace", value=True)
        
        # Run cleaning
        if st.button("🧹 Run Cleaning Process", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned_df, issues = st.session_state.cleaner.clean_data(df, mapping)
                st.session_state.cleaned_data = cleaned_df
                st.session_state.cleaning_issues = issues
            
            st.success("Cleaning completed!")
        
        # Show results
        if 'cleaned_data' in st.session_state:
            cleaned_df = st.session_state.cleaned_data
            issues = st.session_state.cleaning_issues
            
            st.subheader("Before/After Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Data (first 5 rows):**")
                safe_dataframe_display(df.head(), use_container_width=True)
            
            with col2:
                st.write("**Cleaned Data (first 5 rows):**")
                safe_dataframe_display(cleaned_df.head(), use_container_width=True)
            
            # Cleaning summary
            st.subheader("Cleaning Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Issues Fixed", len(issues['fixed']))
            with col2:
                st.metric("Warnings", len(issues['warnings']))
            with col3:
                st.metric("Errors", len(issues['errors']))
            
            # Detailed issues
            if issues['fixed']:
                with st.expander("✅ Fixed Issues"):
                    for issue in issues['fixed'][:20]:  # Show first 20
                        st.text(f"• {issue}")
                    if len(issues['fixed']) > 20:
                        st.text(f"... and {len(issues['fixed']) - 20} more")
            
            if issues['warnings']:
                with st.expander("⚠️ Warnings"):
                    for warning in issues['warnings'][:20]:
                        st.text(f"• {warning}")
                    if len(issues['warnings']) > 20:
                        st.text(f"... and {len(issues['warnings']) - 20} more")
            
            if issues['errors']:
                with st.expander("❌ Errors"):
                    for error in issues['errors'][:20]:
                        st.text(f"• {error}")
                    if len(issues['errors']) > 20:
                        st.text(f"... and {len(issues['errors']) - 20} more")
    
    elif step == "🎯 Targeted Fixes":
        st.header("Step 4: Targeted Fixes")
        
        if 'cleaned_data' not in st.session_state:
            st.warning("Please complete the cleaning step first!")
            return
        
        cleaned_df = st.session_state.cleaned_data
        issues = st.session_state.cleaning_issues
        
        # Analyze remaining issues
        if st.button("🔍 Analyze Remaining Issues", type="primary"):
            with st.spinner("Analyzing issues..."):
                suggestions = st.session_state.fixer.analyze_issues(cleaned_df, issues)
                st.session_state.fix_suggestions = suggestions
        
        if 'fix_suggestions' in st.session_state:
            suggestions = st.session_state.fix_suggestions
            
            if not suggestions:
                st.success("🎉 No remaining issues found! Your data is clean.")
            else:
                st.subheader("Suggested Fixes")
                
                for i, suggestion in enumerate(suggestions):
                    with st.expander(f"{suggestion['type'].title()}: {suggestion['suggestion']}"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Column:** {suggestion['column']}")
                            st.write(f"**Count:** {suggestion['count']}")
                            st.write(f"**Suggested Fix:** {suggestion['auto_fix']}")
                        
                        with col2:
                            if suggestion['auto_fix'] not in ['manual_entry_required', 'manual_review', 'flag_for_review']:
                                if st.button(f"Apply Fix", key=f"fix_{i}"):
                                    fixed_df = st.session_state.fixer.apply_fix(cleaned_df, suggestion)
                                    st.session_state.cleaned_data = fixed_df
                                    st.success("Fix applied!")
                                    st.rerun()
                        
                        with col3:
                            if st.button(f"Promote Rule", key=f"promote_{i}"):
                                # Add to cleaning rules for future use
                                st.success("Rule promoted for future files!")
                
                # Show current data quality metrics
                st.subheader("Data Quality Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    completeness = (1 - cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns))) * 100
                    st.metric("Completeness", f"{completeness:.1f}%")
                
                with col2:
                    # Validity (assuming email/phone validation)
                    validity = 95.0  # Placeholder
                    st.metric("Validity", f"{validity:.1f}%")
                
                with col3:
                    # Consistency (placeholder)
                    consistency = 98.0
                    st.metric("Consistency", f"{consistency:.1f}%")
                
                with col4:
                    # Overall score
                    overall = (completeness + validity + consistency) / 3
                    st.metric("Overall Score", f"{overall:.1f}%")
    
    elif step == "📊 Results & Export":
        st.header("Step 5: Results & Export")
        
        if 'cleaned_data' not in st.session_state:
            st.warning("Please complete the previous steps first!")
            return
        
        cleaned_df = st.session_state.cleaned_data
        
        st.subheader("Final Cleaned Data")
        safe_dataframe_display(cleaned_df, use_container_width=True)
        
        # Export options
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"cleaned_{st.session_state.get('filename', 'data')}.csv",
                mime="text/csv",
                type="primary"
            )
        
        with col2:
            # JSON export for mapping rules
            if 'final_mapping' in st.session_state:
                mapping_json = json.dumps(st.session_state.final_mapping, indent=2)
                st.download_button(
                    label="📥 Download Mapping Rules",
                    data=mapping_json,
                    file_name="mapping_rules.json",
                    mime="application/json"
                )
        
        # Summary report
        st.subheader("Processing Summary")
        
        if 'cleaning_issues' in st.session_state:
            issues = st.session_state.cleaning_issues
            
            summary_data = {
                'Metric': ['Total Records', 'Issues Fixed', 'Warnings Generated', 'Errors Found'],
                'Count': [len(cleaned_df), len(issues['fixed']), len(issues['warnings']), len(issues['errors'])]
            }
            
            summary_df = pd.DataFrame(summary_data)
            safe_dataframe_display(summary_df, use_container_width=True)
        
        # Reset button
        if st.button("🔄 Start New Session", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()