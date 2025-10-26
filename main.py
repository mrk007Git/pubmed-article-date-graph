import osimport os

from pathlib import Pathimport sys

import http.clientimport pyodbc

import xml.etree.ElementTree as ETimport pandas as pd

import pandas as pdimport matplotlib.pyplot as plt

import matplotlib.pyplot as pltimport matplotlib.dates as mdates

import matplotlib.dates as mdatesfrom datetime import datetime

from datetime import datetimefrom dotenv import load_dotenv

from dotenv import load_dotenvimport argparse

import argparse

import time# Load environment variables

import urllib.parseload_dotenv()



# Load environment variablesdef get_database_connection():

load_dotenv()    """Create and return a database connection using the connection string from .env"""

    ado_connection_string = os.getenv('DATABASE_CONNECTION')

# Define search terms for theoretical framework comparison    if not ado_connection_string:

SEARCH_TERMS = [        raise ValueError("DATABASE_CONNECTION not found in environment variables")

    'NASSS Framework',    

    '"Unified Theory of Acceptance and Use of Technology" AND "UTAUT"',    # Parse ADO.NET connection string and convert to ODBC format

    '"Technology Acceptance Model" AND "TAM"',    # Remove quotes if present

    '"Technology Acceptance Model 2" AND "TAM2"',    ado_connection_string = ado_connection_string.strip('"')

    '"UTAUT2"',    

    '"Innovation Diffusion Theory" AND "Rogers"',    # Parse connection string parameters

    '"Task Technology Fit" AND "TTF"'    params = {}

]    for param in ado_connection_string.split(';'):

        if '=' in param:

DATA_DIR = Path('data')            key, value = param.split('=', 1)

DATA_DIR.mkdir(exist_ok=True)            params[key.strip()] = value.strip()

    

def safe_slug(text: str):    # Convert boolean values for ODBC

    slug = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in text)    encrypt = 'yes' if params.get('Encrypt', '').lower() == 'true' else 'no'

    # Collapse repeated underscores    trust_cert = 'yes' if params.get('Trust Server Certificate', '').lower() == 'true' else 'no'

    import re    

    slug = re.sub(r'_+', '_', slug).strip('_')    # Build ODBC connection string

    return slug[:120]    odbc_connection_string = (

        f"DRIVER={{ODBC Driver 17 for SQL Server}};"

def get_api_key():        f"SERVER={params.get('Data Source', '')};"

    """Get API key from environment variables"""        f"DATABASE={params.get('Initial Catalog', '')};"

    api_key = os.getenv('API_KEY')        f"UID={params.get('User ID', '')};"

    if not api_key:        f"PWD={params.get('Password', '')};"

        raise ValueError("API_KEY not found in environment variables")        f"Encrypt={encrypt};"

    return api_key        f"TrustServerCertificate={trust_cert};"

        f"Connection Timeout={params.get('Connect Timeout', '30')}"

def search_pubmed(search_term, retstart=0, retmax=1000):    )

    """Search PubMed for articles matching the search term"""    

    api_key = get_api_key()    print(f"Connecting to: {params.get('Data Source', '')}")

        print(f"Database: {params.get('Initial Catalog', '')}")

    # URL encode the search term    

    encoded_term = urllib.parse.quote(search_term)    try:

            conn = pyodbc.connect(odbc_connection_string)

    conn = http.client.HTTPSConnection("eutils.ncbi.nlm.nih.gov")        print("Successfully connected to database!")

            return conn

    url = f"/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_term}&retstart={retstart}&retmax={retmax}&api_key={api_key}"    except pyodbc.Error as e:

            print(f"Error connecting to database: {e}")

    headers = {        print(f"ODBC Connection String: {odbc_connection_string}")

        'Cookie': 'ncbi_sid=23985CE1D1AB2E25_7370SID'        raise

    }

    def fetch_search_term_name(search_term_id):

    # Caching filename for this slice    """Fetch the search term name from SearchTerm table"""

    cache_file = DATA_DIR / f"esearch_{safe_slug(search_term)}_{retstart}_{retmax}.xml"    query = """

    SELECT [Id], [Term]

    if cache_file.exists():    FROM [dhs].[SearchTerm]

        with cache_file.open('r', encoding='utf-8') as f:    WHERE [Id] = ?

            data = f.read()    """

    else:    

        try:    try:

            conn.request("GET", url, '', headers)        conn = get_database_connection()

            res = conn.getresponse()        df = pd.read_sql_query(query, conn, params=(search_term_id,))

            data = res.read().decode("utf-8")        conn.close()

            with cache_file.open('w', encoding='utf-8') as f:        

                f.write(data)        if len(df) > 0:

        except Exception as e:            term = df['Term'].iloc[0]

            print(f"Error searching PubMed: {e}")            # Remove surrounding quotes if present

            conn.close()            if term.startswith('"') and term.endswith('"'):

            return 0, []                term = term[1:-1]

                    return term

    # Parse XML response        else:

    root = ET.fromstring(data)            return f"Unknown (ID: {search_term_id})"

    except Exception as e:

    # Extract total count and IDs        print(f"Error fetching search term name: {e}")

    count_elem = root.find('.//Count')        return f"Unknown (ID: {search_term_id})"

    total_count = int(count_elem.text) if count_elem is not None and count_elem.text is not None else 0

def fetch_article_data(search_term_id):

    id_list = []    """Fetch article data from the database"""

    for id_elem in root.findall('.//Id'):    query = """

        id_list.append(id_elem.text)    SELECT TOP (1000) [Id]

          ,[SearchTermId]

    conn.close()          ,[PmId]

    return total_count, id_list          ,[DateCompleted]

          ,[DateRevised]

def get_article_details(pmid_list, batch_index=0):          ,[JournalName]

    """Get detailed information for a list of PMIDs"""          ,[ArticleTitle]

    if not pmid_list:          ,[AbstractText]

        return []          ,[IsRelevant]

              ,[Comments]

    api_key = get_api_key()          ,[DateProcessed]

    pmids = ','.join(pmid_list)          ,[EstimatedPercentRelevant]

              ,[AbstractSummary]

    conn = http.client.HTTPSConnection("eutils.ncbi.nlm.nih.gov")          ,[RelevanceReason]

              ,[PromptTokens]

    url = f"/entrez/eutils/esummary.fcgi?db=pubmed&id={pmids}&api_key={api_key}"          ,[CompletionTokens]

              ,[Imported]

    headers = {          ,[SentForProcessing]

        'Cookie': 'ncbi_sid=23985CE1D1AB2E25_7370SID'          ,[DateSentForProcessing]

    }          ,[Included]

              ,[DateIncluded]

    cache_file = DATA_DIR / f"esummary_{batch_index}_{safe_slug(str(len(pmid_list)))}_{safe_slug(pmid_list[0])}.xml"          ,[ShouldBeIncluded]

          ,[ReasonForInclusion]

    if cache_file.exists():          ,[HasFreeTextLink]

        with cache_file.open('r', encoding='utf-8') as f:          ,[FreeTextError]

            data = f.read()          ,[FreeTextLink]

    else:          ,[DOI]

        try:          ,[IncludedInZenodo]

            conn.request("GET", url, '', headers)          ,[DateCreated]

            res = conn.getresponse()    FROM [dhs].[Article]

            data = res.read().decode("utf-8")    WHERE SearchTermId = ?

            with cache_file.open('w', encoding='utf-8') as f:    """

                f.write(data)    

        except Exception as e:    try:

            print(f"Error getting article details: {e}")        conn = get_database_connection()

            conn.close()        df = pd.read_sql_query(query, conn, params=(search_term_id,))

            return []        conn.close()

        return df

    # Parse XML response    except Exception as e:

    root = ET.fromstring(data)        print(f"Error fetching data: {e}")

        raise

    articles = []

    for doc_sum in root.findall('.//DocSum'):def prepare_time_series_data(df):

            article = {}    """Prepare data for time series analysis grouped by quarter"""

                # Convert DateRevised to datetime

            # Get PMID    df['DateRevised'] = pd.to_datetime(df['DateRevised'])

            id_elem = doc_sum.find('.//Id')    

            article['pmid'] = id_elem.text if id_elem is not None else ''    # Remove rows where DateRevised is null

                df_filtered = df.dropna(subset=['DateRevised'])

            # Get publication date (primary)    

            pub_date_elem = doc_sum.find('.//Item[@Name="PubDate"]')    # Sort by DateRevised

            article['pub_date'] = pub_date_elem.text if pub_date_elem is not None else ''    df_filtered = df_filtered.sort_values('DateRevised')

                

            # Get PubMed date from History section (more reliable for indexing date)    # Create quarter grouping

            history_items = doc_sum.findall('.//Item[@Name="History"]/Item[@Name="pubmed"]')    df_filtered['Year'] = df_filtered['DateRevised'].dt.year

            if history_items:    df_filtered['Quarter'] = df_filtered['DateRevised'].dt.quarter

                pubmed_date_elem = history_items[0]    df_filtered['YearQuarter'] = df_filtered['Year'].astype(str) + '-Q' + df_filtered['Quarter'].astype(str)

                article['pubmed_date'] = pubmed_date_elem.text if pubmed_date_elem is not None else ''    

            else:    # Group by quarter and count articles

                article['pubmed_date'] = ''    quarterly_counts = df_filtered.groupby(['Year', 'Quarter']).size().reset_index()

                quarterly_counts.columns = ['Year', 'Quarter', 'ArticleCount']

            # Get EPubDate if available    

            epub_date_elem = doc_sum.find('.//Item[@Name="EPubDate"]')    # Create a proper date for the quarter (using the first day of the quarter)

            article['epub_date'] = epub_date_elem.text if epub_date_elem is not None else ''    quarterly_counts['Date'] = pd.to_datetime(quarterly_counts['Year'].astype(str) + '-' + 

                                                        ((quarterly_counts['Quarter'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01')

            # Get title    

            title_elem = doc_sum.find('.//Item[@Name="Title"]')    # Create quarter label for display

            article['title'] = title_elem.text if title_elem is not None else ''    quarterly_counts['QuarterLabel'] = quarterly_counts['Year'].astype(str) + '-Q' + quarterly_counts['Quarter'].astype(str)

                

            # Get journal    # Create cumulative count

            journal_elem = doc_sum.find('.//Item[@Name="Source"]')    quarterly_counts['CumulativeCount'] = quarterly_counts['ArticleCount'].cumsum()

            article['journal'] = journal_elem.text if journal_elem is not None else ''    

                return quarterly_counts

            # Get DOI if available

            doi_elem = doc_sum.find('.//Item[@Name="DOI"]')def create_growth_chart(data, search_term_id, search_term_name, save_chart=False):

            article['doi'] = doi_elem.text if doi_elem is not None else ''    """Create a graph showing article growth over time by quarter"""

                # Use non-interactive backend if saving

            articles.append(article)    if save_chart:

                plt.switch_backend('Agg')

    conn.close()    

    return articles    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    

def parse_publication_date(article):    # Create title with both ID and term name

    """Parse publication date from article data, prioritizing different date fields"""    title_suffix = f'SearchTerm: "{search_term_name}" (ID: {search_term_id})'

    try:    

        # Priority order: pubmed_date (from History), epub_date, pub_date    # Plot 1: Quarterly article counts

        date_candidates = [    bars = ax1.bar(data['Date'], data['ArticleCount'], alpha=0.7, color='skyblue', width=80)

            article.get('pubmed_date', ''),    ax1.set_title(f'Quarterly Article Counts\n{title_suffix}')

            article.get('epub_date', ''),    ax1.set_xlabel('Year-Quarter')

            article.get('pub_date', '')    ax1.set_ylabel('Number of Articles')

        ]    

            # Format x-axis to show quarters

        for pub_date_str in date_candidates:    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-Q%q'))

            if not pub_date_str:    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

                continue    

                    # Add value labels on bars

            # Remove extra spaces and handle common formats    for i, (bar, count, label) in enumerate(zip(bars, data['ArticleCount'], data['QuarterLabel'])):

            pub_date_str = pub_date_str.strip()        if count > 0:  # Only show label if there are articles

                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 

            # Try different date formats                    str(count), ha='center', va='bottom', fontsize=9)

            formats = [    

                "%Y/%m/%d %H:%M",    # 2025/09/27 06:33 (from History)    # Set x-tick labels manually for better readability

                "%Y/%m/%d",          # 2025/09/27    ax1.set_xticks(data['Date'])

                "%Y %b %d",          # 2025 Aug 29    ax1.set_xticklabels(data['QuarterLabel'], rotation=45, ha='right')

                "%Y %b",             # 2023 Jan    

                "%Y",                # 2023    # Plot 2: Cumulative growth

                "%Y-%m-%d",          # 2023-01-15    ax2.plot(data['Date'], data['CumulativeCount'], marker='o', linewidth=3, 

            ]             markersize=8, color='darkgreen', markerfacecolor='lightgreen', markeredgecolor='darkgreen')

                ax2.set_title(f'Cumulative Article Growth Over Time\n{title_suffix}')

            for fmt in formats:    ax2.set_xlabel('Year-Quarter')

                try:    ax2.set_ylabel('Total Articles')

                    return datetime.strptime(pub_date_str, fmt)    

                except ValueError:    # Format x-axis for cumulative chart

                    continue    ax2.set_xticks(data['Date'])

                ax2.set_xticklabels(data['QuarterLabel'], rotation=45, ha='right')

            # If no format matches, try to extract just the year    

            import re    # Add grid for better readability

            year_match = re.search(r'\b(19|20)\d{2}\b', pub_date_str)    ax1.grid(axis='y', alpha=0.3)

            if year_match:    ax2.grid(axis='y', alpha=0.3)

                year = int(year_match.group())    

                return datetime(year, 1, 1)    plt.tight_layout()

            

        return None    if save_chart:

                # Create safe filename by removing special characters

    except Exception as e:        safe_term_name = "".join(c for c in search_term_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

        print(f"Error parsing dates for article {article.get('pmid', 'unknown')}: {e}")        safe_term_name = safe_term_name.replace(' ', '_')

        return None        filename = f'article_growth_quarterly_{safe_term_name}_id{search_term_id}.png'

        plt.savefig(filename, dpi=300, bbox_inches='tight')

def fetch_all_articles_for_term(search_term, max_results=0):        print(f"Chart saved as: {filename}")

    """Fetch all articles for a search term with paging"""    else:

    print(f"Fetching articles for: {search_term}")        try:

                plt.show()

    all_articles = []        except Exception as e:

    retstart = 0            print(f"Could not display chart: {e}")

    retmax = 1000  # PubMed API allows up to 1000 per request            # Fallback to saving if display fails

                safe_term_name = "".join(c for c in search_term_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

    # First, get total count            safe_term_name = safe_term_name.replace(' ', '_')

    total_count, _ = search_pubmed(search_term, retstart=0, retmax=1)            filename = f'article_growth_quarterly_{safe_term_name}_id{search_term_id}.png'

    print(f"Total articles found: {total_count}")            plt.savefig(filename, dpi=300, bbox_inches='tight')

                print(f"Chart saved as: {filename} (display failed)")

    # Limit to max_results if specified    

    if max_results and max_results > 0 and total_count > max_results:    plt.close()

        total_count = max_results

        print(f"Limiting to first {max_results} articles")def parse_arguments():

        """Parse command line arguments"""

    # Fetch articles in batches    parser = argparse.ArgumentParser(description='Generate article growth charts for PubMed data')

    while retstart < total_count:    parser.add_argument('search_term_id', type=int, 

        current_batch_size = min(retmax, total_count - retstart)                       help='Search Term ID to analyze (e.g., 19)')

        print(f"Fetching batch: {retstart + 1} to {retstart + current_batch_size}")    parser.add_argument('--save', action='store_true', 

                               help='Save the chart as a PNG file instead of displaying')

        # Search for PMIDs    return parser.parse_args()

        _, pmid_list = search_pubmed(search_term, retstart, current_batch_size)

        def main():

        if pmid_list:    """Main function to orchestrate the data fetching and visualization"""

            # Get article details in smaller chunks for eSummary (limit ~200 per request)    try:

            chunk_size = 200        args = parse_arguments()

            for i in range(0, len(pmid_list), chunk_size):        search_term_id = args.search_term_id

                pmid_chunk = pmid_list[i:i + chunk_size]        

                print(f"  Getting details for PMIDs {i+1} to {min(i + chunk_size, len(pmid_list))}")        print(f"Fetching search term name for ID: {search_term_id}...")

                batch_index = (retstart // retmax) * 1000 + i  # pseudo unique batch index        search_term_name = fetch_search_term_name(search_term_id)

                articles = get_article_details(pmid_chunk, batch_index=batch_index)        print(f"Search Term: '{search_term_name}'")

                all_articles.extend(articles)        

                        print(f"Fetching article data for SearchTermId: {search_term_id}...")

                # Rate limiting for eSummary calls        articles_df = fetch_article_data(search_term_id)

                time.sleep(0.34)        

                print(f"Retrieved {len(articles_df)} articles")

        retstart += retmax        

                if len(articles_df) == 0:

        # Rate limiting between search calls            print(f"No articles found for SearchTermId: {search_term_id}")

        time.sleep(0.34)            return None, None

            

    print(f"Retrieved {len(all_articles)} article details")        print("Preparing time series data...")

    return all_articles        time_series_data = prepare_time_series_data(articles_df)

        

def prepare_time_series_data(articles, search_term):        print(f"Time series data prepared with {len(time_series_data)} data points")

    """Prepare articles data for time series analysis by calendar year."""        

    df = pd.DataFrame(articles)        # Display basic statistics

    if df.empty:        print(f"\n=== Data Summary for '{search_term_name}' (ID: {search_term_id}) ===")

        return pd.DataFrame(), search_term        print(f"Quarter range: {time_series_data['QuarterLabel'].iloc[0]} to {time_series_data['QuarterLabel'].iloc[-1]}")

        print(f"Total articles: {time_series_data['CumulativeCount'].max()}")

    # Parse publication/availability dates        print(f"Average articles per quarter: {time_series_data['ArticleCount'].mean():.2f}")

    df['parsed_date'] = [parse_publication_date(row) for row in articles]        print(f"Number of quarters with data: {len(time_series_data)}")

    df_filtered = df.dropna(subset=['parsed_date'])        

    if df_filtered.empty:        # Show sample of the data

        return pd.DataFrame(), search_term        print("\n=== Sample Data (by Quarter) ===")

        print(time_series_data[['QuarterLabel', 'ArticleCount', 'CumulativeCount']].head())

    df_filtered = df_filtered.sort_values('parsed_date')        

    df_filtered['Year'] = df_filtered['parsed_date'].dt.year        print("\nCreating growth chart...")

        create_growth_chart(time_series_data, search_term_id, search_term_name, args.save)

    year_counts = df_filtered.groupby(['Year']).size().reset_index()        

    year_counts.columns = ['Year', 'ArticleCount']        return articles_df, time_series_data

    # Representative date = Jan 1 of that year (good for ordering on axis)        

    year_counts['Date'] = pd.to_datetime(year_counts['Year'].astype(str) + '-01-01')    except Exception as e:

    year_counts['YearLabel'] = year_counts['Year'].astype(str)        print(f"Error in main execution: {e}")

    year_counts = year_counts.sort_values('Date')        return None, None

    year_counts['CumulativeCount'] = year_counts['ArticleCount'].cumsum()

    return year_counts, search_termif __name__ == "__main__":

    articles_data, time_data = main()
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