# PubMed Article Growth Analysis

A Python tool for analyzing and visualizing the growth of research articles over time using the PubMed API. Compare adoption trends across different theoretical frameworks and research domains.

## Features

- **Direct PubMed API Integration**: No database required - fetches live data from PubMed
- **Comparative Analysis**: Compare growth trends across multiple search terms
- **Intelligent Caching**: Saves API responses locally for faster re-runs
- **Publication Date Parsing**: Handles multiple date formats with smart fallbacks
- **Yearly Aggregation**: Groups articles by calendar year for clear trend visualization
- **Dual Chart Output**: Individual term analysis + combined comparison chart

## Prerequisites

1. **Python 3.8+** with virtual environment
2. **PubMed API Key** (free from NCBI)
3. **Required packages** (automatically installed)

## Quick Setup

### 1. Get a PubMed API Key
1. Visit [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/)
2. Sign up for a free NCBI account if needed
3. Generate an API key in your account settings
4. Copy the API key for the next step

### 2. Configure Environment
Create a `.env` file in the project directory:
```bash
API_KEY=your_actual_api_key_here
```

### 3. Install Dependencies
```bash
# Activate your virtual environment (if not already active)
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate   # macOS/Linux

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Run (Save Charts)
```bash
python pubmed_api_analysis.py --save
```

### Quick Test (Limited Results)
```bash
python pubmed_api_analysis.py --save --max-results 50
```

### Generate Individual Charts Too
```bash
python pubmed_api_analysis.py --save --individual
```

### Command Line Options
- `--save`: Save charts as PNG files (recommended when GUI unavailable)
- `--max-results N`: Limit to first N articles per term (0 = unlimited)
- `--individual`: Also generate individual charts per search term

## Output Files

### Charts Generated
- `pubmed_api_combined_cumulative_comparison.png` - Combined growth comparison
- `pubmed_api_growth_[term].png` - Individual term charts (if `--individual` used)

### Cache Directory
- `data/` - Contains cached XML responses for faster re-runs
- Safe to delete to force fresh API calls

## Search Terms

Currently configured to compare these theoretical frameworks:
- **NASSS Framework** - Implementation and evaluation framework
- **UTAUT** - Unified Theory of Acceptance and Use of Technology  
- **TAM** - Technology Acceptance Model
- **TAM2** - Technology Acceptance Model 2
- **UTAUT2** - Extended UTAUT model
- **Innovation Diffusion Theory** - Rogers' diffusion model
- **Task-Technology Fit** - TTF model

## Customization

### Adding New Search Terms
Edit the `SEARCH_TERMS` list in `pubmed_api_analysis.py`:
```python
SEARCH_TERMS = [
    'Your Framework Name',
    '"Exact Phrase Search" AND "Additional Terms"',
    'Broad Term[Title/Abstract]',
]
```

### PubMed Search Syntax
- Use quotes for exact phrases: `"Technology Acceptance Model"`
- Combine with AND/OR: `"TAM" AND "healthcare"`
- Field restrictions: `framework[Title]` or `model[Title/Abstract]`
- See [PubMed Help](https://pubmed.ncbi.nlm.nih.gov/help/) for advanced syntax

## Data Processing

### Date Prioritization
1. **PubMed History Date** (most reliable for indexing)
2. **Electronic Publication Date** 
3. **Primary Publication Date**
4. **Year-only fallback** (extracted from any date field)

### Aggregation
- Groups articles by calendar year
- Calculates both yearly counts and cumulative totals
- Handles missing years gracefully

## Troubleshooting

### Common Issues

**"API_KEY not found"**
- Ensure `.env` file exists in project directory
- Check API key is correctly formatted (no extra spaces/quotes)

**"Error searching PubMed"**
- Verify internet connection
- Check API key is valid
- Try reducing `--max-results` for testing

**Charts not displaying**
- Use `--save` flag to save as PNG files
- Charts auto-save if display fails

### Rate Limiting
- Built-in delays (0.34s between requests)
- NCBI allows higher rates with valid API key
- Caching prevents duplicate API calls

### Cache Management
```bash
# Clear cache to force fresh data
rm -rf data/          # macOS/Linux
Remove-Item data/ -Recurse -Force  # Windows PowerShell
```

## Performance

### Expected Runtime
- **50 articles per term**: ~30 seconds
- **Unlimited** (full dataset): 5-15 minutes depending on term popularity
- **Subsequent runs**: Much faster due to caching

### Memory Usage
- Processes articles in batches (1000 search, 200 details)
- Memory efficient even for large datasets
- XML caching enables restartable processing

## Contributing

### Adding Features
- Granularity options (monthly, quarterly)
- Export to CSV/Excel
- Statistical trend analysis
- Journal/author breakdowns

### Code Structure
- `search_pubmed()`: Handles API search with pagination
- `get_article_details()`: Fetches article metadata in batches  
- `prepare_time_series_data()`: Aggregates by time period
- `create_combined_growth_chart()`: Multi-series visualization

## License

MIT License - feel free to modify and distribute.

## Support

For PubMed API questions: [NCBI E-utilities Help](https://www.ncbi.nlm.nih.gov/books/NBK25501/)

For Python/visualization issues: Check error messages and ensure all dependencies are installed correctly.