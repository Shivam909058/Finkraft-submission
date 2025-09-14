# 🔧 Schema Mapper & Data Quality Fixer

A powerful Streamlit application that automatically maps, cleans, and repairs messy CSV data from different sources into a standardized canonical format.

## 🎯 Problem Statement

Partners send CSVs with different headers for the same concept ("Tax ID," "Reg No.," "VAT#"). Today a data analyst maps columns, cleans values, and reports issues. We want customers to get clean, canonical files without needing a data analyst.

## ✨ Key Features

### 📁 Just Drop Your File
- Simple drag-and-drop file upload
- Instant data preview with statistics
- Support for various CSV formats
- Sample datasets for testing

### 🗺️ Intelligent Schema Mapping
- **95%+ accuracy** with confidence scores
- Fuzzy matching for column name variations
- Visual confidence indicators (🟢 High, 🟡 Medium, 🔴 Low)
- Easy manual overrides for edge cases

### 🧹 One-Click Data Cleaning
- **Date standardization**: Multiple formats → ISO format (YYYY-MM-DD)
- **Currency cleaning**: Remove symbols, commas, quotes
- **Percentage conversion**: "18%" → 0.18 decimal format
- **Email validation**: Format checking and space removal
- **Phone validation**: Pattern matching for various formats
- **Postal code cleaning**: Handle placeholders and formatting

### 📊 Clear Before/After Reports
- Side-by-side data comparison
- Detailed cleaning audit trail
- Categorized issues (Fixed/Warnings/Errors)
- Data quality metrics dashboard

### 🎯 Targeted Fix Suggestions
- **Missing value analysis** with smart fill suggestions
- **Data type inconsistency** detection and fixes
- **Outlier identification** for manual review
- One-click automated fixes where possible

### 🧠 Learning System
- **"Promote this fix"** functionality
- Save mapping rules for future files
- JSON export of learned patterns
- Reusable cleaning configurations

### 💰 Predictable Behavior
- **Deterministic cleaning rules** (no heavy AI costs)
- Transparent processing steps
- Surgical AI only for mapping suggestions
- Complete audit trail for compliance

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy
```

### Installation
1. Clone or download the project files
2. Ensure you have the sample CSV files in the same directory:
   - `Project6InputData1.csv` - Clean data with canonical headers
   - `Project6InputData2.csv` - Messy headers and formats
   - `Project6InputData3.csv` - Different schema with missing columns

### Running the Application
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Usage Workflow

### Step 1: Upload Data
- Choose from sample datasets or upload your own CSV
- Review data preview and column statistics
- Identify data quality issues

### Step 2: Schema Mapping
- Review canonical schema reference
- Generate intelligent mapping suggestions
- Override mappings where needed
- Confirm final column mapping

### Step 3: Clean & Validate
- Configure cleaning options
- Run automated cleaning process
- Review before/after comparison
- Examine detailed cleaning report

### Step 4: Targeted Fixes
- Analyze remaining data quality issues
- Apply one-click fixes for common problems
- Flag complex issues for manual review
- Monitor data quality metrics

### Step 5: Results & Export
- Download cleaned CSV file
- Export mapping rules for reuse
- Review processing summary
- Start new session if needed

## 📊 Sample Datasets

### Project6InputData1.csv (Clean)
- Canonical column headers
- Consistent data formats
- Minimal cleaning required
- **Use case**: Baseline for comparison

### Project6InputData2.csv (Messy)
- Mixed date formats: `23/06/2025`, `25-Mar-2025`
- Currency symbols: `₹`, `Rs`, `"4,999.00"`
- Percentage formats: `18%`, `0%`
- Header variations: `Order No`, `E-mail`, `Phone #`
- **Use case**: Typical messy partner data

### Project6InputData3.csv (Different Schema)
- Completely different column names
- Missing standard columns
- Postal codes with placeholders (`667XX2`)
- Alternative date format: `02 Aug 2025`
- **Use case**: New partner with unique format

## 🔧 Technical Architecture

### Core Components

#### SchemaMapper Class
- Fuzzy string matching for column names
- Confidence scoring algorithm
- Predefined mapping rules for common variations
- Support for manual overrides

#### DataCleaner Class
- Modular cleaning functions by data type
- Configurable cleaning rules
- Detailed issue tracking and reporting
- Safe data type conversions

#### TargetedFixer Class
- Statistical analysis for outlier detection
- Missing value pattern analysis
- Automated fix suggestions
- Rule promotion for learning

### Data Flow

Raw CSV → Schema Mapping → Data Cleaning → Targeted Fixes → Clean Output
↓ ↓ ↓ ↓ ↓
Preview Confidence Issue Tracking Analytics Export


## 🎨 User Interface

### Navigation
- **Sidebar navigation** with 5-step workflow
- **Progress tracking** through the process
- **Clear visual indicators** for data quality

### Data Visualization
- **Interactive data tables** with sorting/filtering
- **Confidence score indicators** with color coding
- **Before/after comparisons** side-by-side
- **Quality metrics dashboard** with percentages

## 📈 Data Quality Metrics

The app tracks and displays:
- **Completeness**: Percentage of non-null values
- **Validity**: Percentage of values passing validation
- **Consistency**: Percentage of standardized formats
- **Overall Score**: Weighted average of all metrics

## 🔄 Reusability Features

### Mapping Rules Export
```json
{
  "Order No": "order_id",
  "Customer": "customer_name",
  "E-mail": "email",
  "Unit Price": "unit_price"
}
```

### Cleaning Rules Persistence
- Date format preferences
- Currency symbol handling
- Validation patterns
- Custom cleaning functions

## 🛠️ Customization

### Adding New Canonical Fields
Edit the `CANONICAL_SCHEMA` dictionary in `main.py`:
```python
CANONICAL_SCHEMA = {
    'new_field': {
        'description': 'Field description',
        'example': 'Example value'
    }
}
```

### Custom Cleaning Rules
Extend the `DataCleaner` class with new cleaning methods:
```python
def _clean_custom_field(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
    # Custom cleaning logic
    return cleaned_series, issues
```

### Mapping Rule Extensions
Add new variations to `MAPPING_RULES`:
```python
MAPPING_RULES = {
    'field_name': ['variation1', 'variation2', 'variation3']
}
```

## 🚨 Error Handling

The application includes robust error handling for:
- **File format issues**: Invalid CSV, encoding problems
- **Data type conflicts**: Mixed types in columns
- **Memory constraints**: Large file handling
- **Display issues**: Arrow serialization problems

## 📝 Logging and Audit Trail

Every operation is logged with:
- **Timestamp** of the operation
- **Original and cleaned values**
- **Confidence scores** for mappings
- **Applied transformations**
- **User decisions** and overrides

## 🔒 Data Privacy

- **No data persistence**: Files are processed in memory only
- **Local processing**: No data sent to external services
- **Session isolation**: Each user session is independent
- **Secure cleanup**: Session data cleared on completion

## 🎯 Use Cases

### Enterprise Data Integration
- Standardize partner data feeds
- Automate vendor file processing
- Ensure consistent data quality

### Data Migration Projects
- Clean legacy system exports
- Standardize formats across systems
- Validate data before migration

### Compliance and Reporting
- Ensure data quality standards
- Generate audit trails
- Maintain data lineage


**Built with Skills using Streamlit, Pandas, and Python**

*Transform your messy data into clean, standardized formats with confidence.*