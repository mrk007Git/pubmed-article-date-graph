import os
from pathlib import Path
import http.client
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from dotenv import load_dotenv
import argparse
import time
import urllib.parse

# Load environment variables
load_dotenv()

# Define search terms for theoretical framework comparison
SEARCH_TERMS = [
    'NASSS Framework',
    '"Unified Theory of Acceptance and Use of Technology" AND "UTAUT"',
    '"Technology Acceptance Model" AND "TAM"',
    '"Technology Acceptance Model 2" AND "TAM2"'
]

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

def safe_slug(text: str):
    slug = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in text)
    # Collapse repeated underscores
    import re
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug[:120]

def get_api_key():
    """Get API key from environment variables"""
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("API_KEY not found in environment variables")
    return api_key

def search_pubmed(search_term, retstart=0, retmax=1000):
    """Search PubMed for articles matching the search term"""
    api_key = get_api_key()
    
    # URL encode the search term
    encoded_term = urllib.parse.quote(search_term)
    
    conn = http.client.HTTPSConnection("eutils.ncbi.nlm.nih.gov")
    
    url = f"/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_term}&retstart={retstart}&retmax={retmax}&api_key={api_key}"
    
    headers = {
        'Cookie': 'ncbi_sid=23985CE1D1AB2E25_7370SID'
    }
    
    # Caching filename for this slice
    cache_file = DATA_DIR / f"esearch_{safe_slug(search_term)}_{retstart}_{retmax}.xml"

    if cache_file.exists():
        with cache_file.open('r', encoding='utf-8') as f:
            data = f.read()
    else:
        try:
            conn.request("GET", url, '', headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            with cache_file.open('w', encoding='utf-8') as f:
                f.write(data)
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            conn.close()
            return 0, []
        
    # Parse XML response
    root = ET.fromstring(data)

    # Extract total count and IDs
    count_elem = root.find('.//Count')
    total_count = int(count_elem.text) if count_elem is not None and count_elem.text is not None else 0

    id_list = []
    for id_elem in root.findall('.//Id'):
        id_list.append(id_elem.text)

    conn.close()
    return total_count, id_list

def get_article_details(pmid_list, batch_index=0):
    """Get detailed information for a list of PMIDs"""
    if not pmid_list:
        return []
    
    api_key = get_api_key()
    pmids = ','.join(pmid_list)
    
    conn = http.client.HTTPSConnection("eutils.ncbi.nlm.nih.gov")
    
    url = f"/entrez/eutils/esummary.fcgi?db=pubmed&id={pmids}&api_key={api_key}"
    
    headers = {
        'Cookie': 'ncbi_sid=23985CE1D1AB2E25_7370SID'
    }
    
    cache_file = DATA_DIR / f"esummary_{batch_index}_{safe_slug(str(len(pmid_list)))}_{safe_slug(pmid_list[0])}.xml"

    if cache_file.exists():
        with cache_file.open('r', encoding='utf-8') as f:
            data = f.read()
    else:
        try:
            conn.request("GET", url, '', headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            with cache_file.open('w', encoding='utf-8') as f:
                f.write(data)
        except Exception as e:
            print(f"Error getting article details: {e}")
            conn.close()
            return []

    # Parse XML response
    root = ET.fromstring(data)

    articles = []
    for doc_sum in root.findall('.//DocSum'):
            article = {}
            
            # Get PMID
            id_elem = doc_sum.find('.//Id')
            article['pmid'] = id_elem.text if id_elem is not None else ''
            
            # Get publication date (primary)
            pub_date_elem = doc_sum.find('.//Item[@Name="PubDate"]')
            article['pub_date'] = pub_date_elem.text if pub_date_elem is not None else ''
            
            # Get PubMed date from History section (more reliable for indexing date)
            history_items = doc_sum.findall('.//Item[@Name="History"]/Item[@Name="pubmed"]')
            if history_items:
                pubmed_date_elem = history_items[0]
                article['pubmed_date'] = pubmed_date_elem.text if pubmed_date_elem is not None else ''
            else:
                article['pubmed_date'] = ''
            
            # Get EPubDate if available
            epub_date_elem = doc_sum.find('.//Item[@Name="EPubDate"]')
            article['epub_date'] = epub_date_elem.text if epub_date_elem is not None else ''
            
            # Get title
            title_elem = doc_sum.find('.//Item[@Name="Title"]')
            article['title'] = title_elem.text if title_elem is not None else ''
            
            # Get journal
            journal_elem = doc_sum.find('.//Item[@Name="Source"]')
            article['journal'] = journal_elem.text if journal_elem is not None else ''
            
            # Get DOI if available
            doi_elem = doc_sum.find('.//Item[@Name="DOI"]')
            article['doi'] = doi_elem.text if doi_elem is not None else ''
            
            articles.append(article)
        
    conn.close()
    return articles

def parse_publication_date(article):
    """Parse publication date from article data, prioritizing different date fields"""
    try:
        # Priority order: pubmed_date (from History), epub_date, pub_date
        date_candidates = [
            article.get('pubmed_date', ''),
            article.get('epub_date', ''),
            article.get('pub_date', '')
        ]
        
        for pub_date_str in date_candidates:
            if not pub_date_str:
                continue
                
            # Remove extra spaces and handle common formats
            pub_date_str = pub_date_str.strip()
            
            # Try different date formats
            formats = [
                "%Y/%m/%d %H:%M",    # 2025/09/27 06:33 (from History)
                "%Y/%m/%d",          # 2025/09/27
                "%Y %b %d",          # 2025 Aug 29
                "%Y %b",             # 2023 Jan
                "%Y",                # 2023
                "%Y-%m-%d",          # 2023-01-15
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(pub_date_str, fmt)
                except ValueError:
                    continue
            
            # If no format matches, try to extract just the year
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', pub_date_str)
            if year_match:
                year = int(year_match.group())
                return datetime(year, 1, 1)
        
        return None
        
    except Exception as e:
        print(f"Error parsing dates for article {article.get('pmid', 'unknown')}: {e}")
        return None

def fetch_all_articles_for_term(search_term, max_results=0):
    """Fetch all articles for a search term with paging"""
    print(f"Fetching articles for: {search_term}")
    
    all_articles = []
    retstart = 0
    retmax = 1000  # PubMed API allows up to 1000 per request
    
    # First, get total count
    total_count, _ = search_pubmed(search_term, retstart=0, retmax=1)
    print(f"Total articles found: {total_count}")
    
    # Limit to max_results if specified
    if max_results and max_results > 0 and total_count > max_results:
        total_count = max_results
        print(f"Limiting to first {max_results} articles")
    
    # Fetch articles in batches
    while retstart < total_count:
        current_batch_size = min(retmax, total_count - retstart)
        print(f"Fetching batch: {retstart + 1} to {retstart + current_batch_size}")
        
        # Search for PMIDs
        _, pmid_list = search_pubmed(search_term, retstart, current_batch_size)
        
        if pmid_list:
            # Get article details in smaller chunks for eSummary (limit ~200 per request)
            chunk_size = 200
            for i in range(0, len(pmid_list), chunk_size):
                pmid_chunk = pmid_list[i:i + chunk_size]
                print(f"  Getting details for PMIDs {i+1} to {min(i + chunk_size, len(pmid_list))}")
                batch_index = (retstart // retmax) * 1000 + i  # pseudo unique batch index
                articles = get_article_details(pmid_chunk, batch_index=batch_index)
                all_articles.extend(articles)
                
                # Rate limiting for eSummary calls
                time.sleep(0.34)
        
        retstart += retmax
        
        # Rate limiting between search calls
        time.sleep(0.34)
    
    print(f"Retrieved {len(all_articles)} article details")
    return all_articles

def prepare_time_series_data(articles, search_term):
    """Prepare articles data for time series analysis by calendar year."""
    df = pd.DataFrame(articles)
    if df.empty:
        return pd.DataFrame(), search_term

    # Parse publication/availability dates
    df['parsed_date'] = [parse_publication_date(row) for row in articles]
    df_filtered = df.dropna(subset=['parsed_date'])
    if df_filtered.empty:
        return pd.DataFrame(), search_term

    df_filtered = df_filtered.sort_values('parsed_date')
    df_filtered['Year'] = df_filtered['parsed_date'].dt.year

    year_counts = df_filtered.groupby(['Year']).size().reset_index()
    year_counts.columns = ['Year', 'ArticleCount']
    # Representative date = Jan 1 of that year (good for ordering on axis)
    year_counts['Date'] = pd.to_datetime(year_counts['Year'].astype(str) + '-01-01')
    year_counts['YearLabel'] = year_counts['Year'].astype(str)
    year_counts = year_counts.sort_values('Date')
    year_counts['CumulativeCount'] = year_counts['ArticleCount'].cumsum()
    return year_counts, search_term

def create_growth_chart(data, search_term, save_chart=False):
    """Create a graph showing article growth over time by year."""
    if data.empty:
        print(f"No data to plot for: {search_term}")
        return
    
    # Use non-interactive backend if saving
    if save_chart:
        plt.switch_backend('Agg')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Create title with search term
    title_suffix = f'Search Term: "{search_term}"'
    
    # Plot 1: Yearly article counts
    bars = ax1.bar(data['Date'], data['ArticleCount'], alpha=0.7, color='skyblue', width=200)
    ax1.set_title(f'Yearly Article Counts - PubMed API\n{title_suffix}')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Articles')
    
    # Add value labels on bars
    label_field = 'YearLabel' if 'YearLabel' in data.columns else ('HalfLabel' if 'HalfLabel' in data.columns else ('QuarterLabel' if 'QuarterLabel' in data.columns else None))
    for i, (bar, count, label) in enumerate(zip(bars, data['ArticleCount'], data[label_field])):
        if count > 0:  # Only show label if there are articles
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=9)
    
    # Set x-tick labels manually for better readability
    ax1.set_xticks(data['Date'])
    ax1.set_xticklabels(data[label_field], rotation=45, ha='right')
    
    # Plot 2: Cumulative growth
    ax2.plot(data['Date'], data['CumulativeCount'], marker='o', linewidth=3, 
             markersize=8, color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen')
    ax2.set_title(f'Cumulative Article Growth Over Time (Yearly) - PubMed API\n{title_suffix}')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Articles')
    
    # Format x-axis for cumulative chart
    ax2.set_xticks(data['Date'])
    ax2.set_xticklabels(data[label_field], rotation=45, ha='right')
    
    # Add grid for better readability
    ax1.grid(axis='y', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_chart:
        # Create safe filename
        safe_term_name = "".join(c for c in search_term if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_term_name = safe_term_name.replace(' ', '_').replace('"', '')
        filename = f'pubmed_api_growth_{safe_term_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved as: {filename}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display chart: {e}")
            # Fallback to saving if display fails
            safe_term_name = "".join(c for c in search_term if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_term_name = safe_term_name.replace(' ', '_').replace('"', '')
            filename = f'pubmed_api_growth_{safe_term_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Chart saved as: {filename} (display failed)")
    
    plt.close()

def create_combined_growth_chart(datasets, save_chart=False, filename_prefix='pubmed_api_combined'):
    """Create a single cumulative time series comparison chart (yearly) for multiple search terms.

    datasets: list of tuples (search_term, dataframe) where dataframe has columns:
        Date, CumulativeCount, YearLabel (or HalfLabel/QuarterLabel fallback)
    """
    if not datasets:
        print("No datasets provided for combined chart.")
        return

    if save_chart:
        plt.switch_backend('Agg')

    plt.figure(figsize=(15, 8))
    # Get a qualitative colormap for up to 10 distinct colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(cmap.N)]

    # Determine global date range for x-axis ordering
    all_dates = set()
    for _, df in datasets:
        all_dates.update(df['Date'].tolist())
    ordered_dates = sorted(all_dates)

    # Map date to an index for aligned plotting (in case some series miss quarters)
    date_to_index = {d: i for i, d in enumerate(ordered_dates)}

    for idx, (term, df) in enumerate(datasets):
        # Ensure dataframe is sorted
        df = df.sort_values('Date')
        x = [date_to_index[d] for d in df['Date']]
        y = df['CumulativeCount']
        label = term
        plt.plot(x, y, marker='o', linewidth=2, markersize=5, color=colors[idx % len(colors)], label=label)

    # Build x-axis tick labels using ordered_dates converted to year labels (fallbacks supported)
    period_labels_map = {}
    for _, df in datasets:
        col = 'YearLabel' if 'YearLabel' in df.columns else ('HalfLabel' if 'HalfLabel' in df.columns else ('QuarterLabel' if 'QuarterLabel' in df.columns else None))
        if col is None:
            continue
        for d, plabel in zip(df['Date'], df[col]):
            period_labels_map[d] = plabel
    x_ticks = list(range(len(ordered_dates)))
    x_tick_labels = [period_labels_map.get(d, d.strftime('%Y')) for d in ordered_dates]

    plt.xticks(x_ticks, x_tick_labels, rotation=45, ha='right')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Articles')
    plt.title('Cumulative Article Growth Comparison (Yearly, PubMed API)')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Search Terms', fontsize=9)
    plt.tight_layout()

    if save_chart:
        filename = f"{filename_prefix}_cumulative_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Combined chart saved as: {filename}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display combined chart: {e}")
            filename = f"{filename_prefix}_cumulative_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Combined chart saved as: {filename} (display failed)")
    plt.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate combined article growth chart for predefined PubMed search terms')
    parser.add_argument('--save', action='store_true', 
                       help='Save charts as PNG files instead of displaying')
    parser.add_argument('--max-results', type=int, default=0,
                       help='Maximum results per term (0 = unlimited)')
    parser.add_argument('--individual', action='store_true',
                       help='Also generate individual charts per term')
    return parser.parse_args()

def main():
    """Main function to orchestrate the PubMed API data fetching and visualization"""
    try:
        args = parse_arguments()
        
        # Always process predefined search terms
        search_terms = SEARCH_TERMS
        print(f"Processing predefined search terms: {len(search_terms)} terms")
        
        combined_datasets = []  # collect (term, df) for combined chart

        # Process each search term
        for search_term in search_terms:
            print(f"\n{'='*60}")
            print(f"Processing: {search_term}")
            print('='*60)
            
            # Fetch articles from PubMed API
            articles = fetch_all_articles_for_term(search_term, args.max_results)
            
            if not articles:
                print(f"No articles found for: {search_term}")
                continue
            
            # Prepare time series data
            time_series_data, clean_term = prepare_time_series_data(articles, search_term)
            
            if time_series_data.empty:
                print(f"No valid publication dates found for: {search_term}")
                continue
            
            # Display statistics
            print(f"\n=== Data Summary for '{search_term}' ===")
            range_col = 'YearLabel' if 'YearLabel' in time_series_data.columns else (
                'HalfLabel' if 'HalfLabel' in time_series_data.columns else (
                    'QuarterLabel' if 'QuarterLabel' in time_series_data.columns else None))
            if range_col is None:
                print("No period label column available.")
            else:
                print(f"Year range: {time_series_data[range_col].iloc[0]} to {time_series_data[range_col].iloc[-1]}")
            print(f"Total articles: {time_series_data['CumulativeCount'].max()}")
            print(f"Average articles per year: {time_series_data['ArticleCount'].mean():.2f}")
            print(f"Number of years with data: {len(time_series_data)}")

            # Show sample data
            if range_col:
                print("\n=== Sample Data (by Year) ===")
                print(time_series_data[[range_col, 'ArticleCount', 'CumulativeCount']].head())

            # Store for combined chart
            store_cols = ['Date','YearLabel','CumulativeCount'] if 'YearLabel' in time_series_data.columns else (
                ['Date','HalfLabel','CumulativeCount'] if 'HalfLabel' in time_series_data.columns else ['Date','QuarterLabel','CumulativeCount'])
            combined_datasets.append((search_term, time_series_data[store_cols].copy()))

            if args.individual:
                # Create individual chart
                print("\nCreating growth chart...")
                create_growth_chart(time_series_data, search_term, args.save)
            
            # Add delay between different search terms to respect rate limits
            if len(search_terms) > 1:
                print("Waiting 2 seconds before next search term...")
                time.sleep(2)
        
        # Always create combined chart
        if combined_datasets:
            print("\nGenerating combined cumulative comparison chart for all terms...")
            create_combined_growth_chart(combined_datasets, save_chart=args.save)

        print(f"\n{'='*60}")
        print("Analysis complete!")
        print('='*60)
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()