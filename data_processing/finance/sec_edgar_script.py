import pandas as pd
import json
import sys
import argparse
import subprocess
import re
from datetime import datetime
from __init__ import DATASET_DIR

def fetch_data():
    """
    Fetches S&P 500 component stocks data from Wikipedia and returns a DataFrame
    with Symbol, Security, GICS Sector, GICS Sub-Industry, and CIK.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    
    # The S&P 500 components table is typically the first table on the page
    sp500_table = tables[0]
    
    # Check if the table has the expected columns
    expected_columns = ['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'CIK']
    if not all(col in sp500_table.columns for col in expected_columns):
        # If the expected columns aren't found, try to map them
        columns_map = {}
        for col in sp500_table.columns:
            col_lower = str(col).lower()
            if 'ticker' in col_lower or 'symbol' in col_lower:
                columns_map['Symbol'] = col
            elif 'company' in col_lower or 'security' in col_lower or 'name' in col_lower:
                columns_map['Security'] = col
            elif 'gics sector' in col_lower or 'sector' in col_lower:
                columns_map['GICS_Sector'] = col
            elif 'gics sub-industry' in col_lower or 'sub-industry' in col_lower or 'industry' in col_lower:
                columns_map['GICS_SubIndustry'] = col
            elif 'cik' in col_lower:
                columns_map['CIK'] = col
        
        # Create result dataframe with standardized column names
        result_df = pd.DataFrame()
        result_df['Symbol'] = sp500_table[columns_map.get('Symbol')].astype(str).str.strip()
        result_df['Security'] = sp500_table[columns_map.get('Security')].astype(str).str.strip()
        result_df['GICS_Sector'] = sp500_table[columns_map.get('GICS_Sector')].astype(str).str.strip()
        result_df['GICS_SubIndustry'] = sp500_table[columns_map.get('GICS_SubIndustry')].astype(str).str.strip()
        
        # Handle CIK - ensure it's properly formatted
        if 'CIK' in columns_map:
            # Convert to string, strip, and ensure proper formatting (10-digit with leading zeros)
            result_df['CIK'] = sp500_table[columns_map.get('CIK')].astype(str).str.strip()
            # Pad with leading zeros to ensure 10 digits
            result_df['CIK'] = result_df['CIK'].apply(lambda x: x.zfill(10) if x.isdigit() else x)
        else:
            # If CIK column not found, add an empty column
            result_df['CIK'] = ""
            print("Warning: CIK column not found in the Wikipedia table")
    else:
        # If the table already has the expected columns, use them directly
        result_df = pd.DataFrame()
        result_df['Symbol'] = sp500_table['Symbol'].astype(str).str.strip()
        result_df['Security'] = sp500_table['Security'].astype(str).str.strip()
        result_df['GICS_Sector'] = sp500_table['GICS Sector'].astype(str).str.strip()
        result_df['GICS_SubIndustry'] = sp500_table['GICS Sub-Industry'].astype(str).str.strip()
        result_df['CIK'] = sp500_table['CIK'].astype(str).str.strip()
        # Ensure CIK is properly formatted (10 digits with leading zeros)
        result_df['CIK'] = result_df['CIK'].apply(lambda x: x.zfill(10) if x.isdigit() else x)
    
    print(f"Successfully retrieved {len(result_df)} S&P 500 companies")
    return result_df

def filter_companies(df, args):
    filtered_df = df.copy()
    
    # Filter by sector if specified
    if args.sectors:
        # Convert both to lowercase for case-insensitive matching
        sectors_lower = [s.lower() for s in args.sectors]
        filtered_df = filtered_df[filtered_df['GICS_Sector'].str.lower().isin(sectors_lower)]
        print(f"Filtered to {len(filtered_df)} companies in sectors: {', '.join(args.sectors)}")
    
    # Filter by sub-industry if specified
    if args.industries:
        industries_lower = [i.lower() for i in args.industries]
        filtered_df = filtered_df[filtered_df['GICS_SubIndustry'].str.lower().isin(industries_lower)]
        print(f"Filtered to {len(filtered_df)} companies in industries: {', '.join(args.industries)}")
    
    # Filter by symbol regex if specified
    if args.symbol_regex:
        pattern = re.compile(args.symbol_regex, re.IGNORECASE)
        filtered_df = filtered_df[filtered_df['Symbol'].apply(lambda x: bool(pattern.search(x)))]
        print(f"Filtered to {len(filtered_df)} companies matching symbol pattern: {args.symbol_regex}")
    
    # Filter by name regex if specified
    if args.name_regex:
        pattern = re.compile(args.name_regex, re.IGNORECASE)
        filtered_df = filtered_df[filtered_df['Name'].apply(lambda x: bool(pattern.search(x)))]
        print(f"Filtered to {len(filtered_df)} companies matching name pattern: {args.name_regex}")
    
    # Get random sample if specified
    if args.random:
        filtered_df = filtered_df.sample(min(args.random, len(filtered_df)))
        print(f"Selected {len(filtered_df)} companies randomly")
    
    # Limit to top N companies if specified
    if args.limit and len(filtered_df) > args.limit:
        filtered_df = filtered_df.head(args.limit)
        print(f"Limited to the first {args.limit} companies")
    
    # Check if we have any companies left after filtering
    if len(filtered_df) == 0:
        print("Warning: No companies match the selected filters!")
        print("Using all S&P 500 companies instead.")
        return df
        
    return filtered_df

def save_to_csv(df, filename=f'{DATASET_DIR}/sp500_companies.csv'):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return filename

def print_available_sectors_and_industries(df):
    print("\nAvailable GICS Sectors:")
    for sector in sorted(df['GICS_Sector'].unique()):
        print(f"  - {sector}")
    
    print("\nAvailable GICS Sub-Industries:")
    # Get a sample of industries for each sector to avoid overwhelming output
    for sector in sorted(df['GICS_Sector'].unique()):
        sector_industries = df[df['GICS_Sector'] == sector]['GICS_SubIndustry'].unique()
        print(f"  {sector} ({len(sector_industries)} sub-industries):")
        # Print at most 3 examples per sector
        for industry in sorted(sector_industries)[:3]:
            print(f"    - {industry}")
        if len(sector_industries) > 3:
            print(f"    - ... and {len(sector_industries) - 3} more")

def update_config(symbols, config_file=f'{DATASET_DIR}/config.json', args=None):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        config['download_filings']['cik_tickers'] = symbols.tolist()
        
        if args:
            current_year = datetime.now().year
            
            if args.auto_year:
                # Use current year for both start and end
                config['download_filings']['start_year'] = current_year
                config['download_filings']['end_year'] = current_year
            else:
                # Use provided years or keep existing values
                if args.start_year:
                    config['download_filings']['start_year'] = args.start_year
                if args.end_year:
                    config['download_filings']['end_year'] = args.end_year
            
            # Update quarters if provided
            if args.quarters:
                config['download_filings']['quarters'] = args.quarters
            
            # Update filing types if provided
            if args.filing_types:
                config['download_filings']['filing_types'] = args.filing_types
                config['extract_items']['filing_types'] = args.filing_types
            
            # Update user agent if provided
            if args.user_agent:
                config['download_filings']['user_agent'] = args.user_agent
                
            # Update folders if provided
            if args.raw_folder:
                config['download_filings']['raw_filings_folder'] = args.raw_folder
                config['extract_items']['raw_filings_folder'] = args.raw_folder
                
            if args.extracted_folder:
                config['extract_items']['extracted_filings_folder'] = args.extracted_folder
                
            if args.indices_folder:
                config['download_filings']['indices_folder'] = args.indices_folder
                
            if args.metadata_file:
                config['download_filings']['filings_metadata_file'] = args.metadata_file
                config['extract_items']['filings_metadata_file'] = args.metadata_file
                
            # Update boolean flags
            if args.skip_present_indices is not None:
                config['download_filings']['skip_present_indices'] = args.skip_present_indices
                
            if args.skip_extracted_filings is not None:
                config['extract_items']['skip_extracted_filings'] = args.skip_extracted_filings
                
            if args.include_signature is not None:
                config['extract_items']['include_signature'] = args.include_signature
                
            if args.remove_tables is not None:
                config['extract_items']['remove_tables'] = args.remove_tables
        
        # Write the updated config back to file
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Updated config.json with {len(symbols)} ticker symbols and provided parameters")
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Creating new file...")
        
        # Get current year
        current_year = datetime.now().year
        
        # Default values
        start_year = current_year
        end_year = current_year
        
        # Override with args if provided
        if args:
            if not args.auto_year:
                if args.start_year:
                    start_year = args.start_year
                if args.end_year:
                    end_year = args.end_year
        
        # Create a new config file
        config = {
            "download_filings": {
                "start_year": start_year,
                "end_year": end_year,
                "quarters": args.quarters if args and args.quarters else [1, 2, 3, 4],
                "filing_types": args.filing_types if args and args.filing_types else ["10-K", "10-Q", "8-K"],
                "cik_tickers": symbols.tolist(),
                "user_agent": args.user_agent if args and args.user_agent else "Yixiao Zeng (yixiaozeng0208@outlook.com)",
                "raw_filings_folder": args.raw_folder if args and args.raw_folder else "sec_edgar_raw_filings",
                "indices_folder": args.indices_folder if args and args.indices_folder else "sec_edgar_indices",
                "filings_metadata_file": args.metadata_file if args and args.metadata_file else "sec_edgar_filings_metadata.csv",
                "skip_present_indices": args.skip_present_indices if args and args.skip_present_indices is not None else True
            },
            "extract_items": {
                "raw_filings_folder": args.raw_folder if args and args.raw_folder else "sec_edgar_raw_filings",
                "extracted_filings_folder": args.extracted_folder if args and args.extracted_folder else "sec_edgar_extracted_filings",
                "filings_metadata_file": args.metadata_file if args and args.metadata_file else "sec_edgar_filings_metadata.csv",
                "filing_types": args.filing_types if args and args.filing_types else ["10-K", "10-Q", "8-K"],
                "include_signature": args.include_signature if args and args.include_signature is not None else False,
                "items_to_extract": [],
                "remove_tables": args.remove_tables if args and args.remove_tables is not None else True,
                "skip_extracted_filings": args.skip_extracted_filings if args and args.skip_extracted_filings is not None else True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Created config.json with {len(symbols)} ticker symbols and provided parameters")
    
    return config

def run_scripts(args):
    scripts_to_run = []
    
    if args.download:
        scripts_to_run.append('download_filings.py')
    
    if args.extract:
        scripts_to_run.append('extract_items.py')
    
    for script in scripts_to_run:
        print(f"\nRunning {script}...")
        try:
            subprocess.run([sys.executable, script], check=True)
            print(f"Successfully executed {script}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")
        except FileNotFoundError:
            print(f"Error: {script} not found in the current directory")

def parse_arguments():
    parser = argparse.ArgumentParser(description='S&P 500 SEC Filing Automation Tool')
    
    # CSV output
    parser.add_argument('--csv', type=str, default=f'{DATASET_DIR}/sp500_companies.csv',
                        help='Output CSV filename (default: sp500_companies.csv)')
    
    # Config file
    parser.add_argument('--config', type=str, default=f'{DATASET_DIR}/config.json',
                        help='Config JSON filename (default: config.json)')
    
    # Company selection options
    selection_group = parser.add_argument_group('Company Selection Options')
    selection_group.add_argument('--sectors', type=str, nargs='+',
                               help='Filter companies by GICS Sectors (e.g., --sectors "Information Technology" "Health Care")')
    selection_group.add_argument('--industries', type=str, nargs='+',
                               help='Filter companies by GICS Sub-Industries (e.g., --industries "Semiconductors" "Software")')
    selection_group.add_argument('--symbol-regex', type=str,
                               help='Filter companies by symbol regex (e.g., --symbol-regex "^A.*")')
    selection_group.add_argument('--name-regex', type=str,
                               help='Filter companies by company name regex (e.g., --name-regex "Tech|Software")')
    selection_group.add_argument('--limit', type=int,
                               help='Limit to the first N companies after other filters')
    selection_group.add_argument('--random', type=int,
                               help='Select N random companies after other filters')
    selection_group.add_argument('--list-sectors', action='store_true',
                               help='Print available sectors and industries and exit')
    
    # Year settings
    year_group = parser.add_mutually_exclusive_group()
    year_group.add_argument('--auto-year', action='store_true',
                          help='Automatically use current year for start_year and end_year')
    year_group.add_argument('--start-year', type=int,
                          help='Set the start year for filings download')
    parser.add_argument('--end-year', type=int,
                      help='Set the end year for filings download')
    
    # Filing parameters
    parser.add_argument('--quarters', type=int, nargs='+',
                      help='Quarters to download (e.g., --quarters 1 2 3 4)')
    parser.add_argument('--filing-types', type=str, nargs='+',
                      help='Filing types to download (e.g., --filing-types 10-K 10-Q 8-K)')
    parser.add_argument('--user-agent', type=str,
                      help='User agent for SEC EDGAR requests (e.g., "Your Name your.email@example.com")')
    
    # Folder paths
    parser.add_argument('--raw-folder', type=str,
                      help='Folder for raw filings')
    parser.add_argument('--extracted-folder', type=str,
                      help='Folder for extracted filings')
    parser.add_argument('--indices-folder', type=str,
                      help='Folder for indices')
    parser.add_argument('--metadata-file', type=str,
                      help='Path to metadata CSV file')
    
    # Boolean flags
    parser.add_argument('--skip-present-indices', action='store_true', dest='skip_present_indices',
                      help='Skip downloading indices that are already present')
    parser.add_argument('--no-skip-present-indices', action='store_false', dest='skip_present_indices',
                      help='Do not skip downloading indices that are already present')
    parser.set_defaults(skip_present_indices=None)
    
    parser.add_argument('--skip-extracted-filings', action='store_true', dest='skip_extracted_filings',
                      help='Skip extracting from filings that have already been processed')
    parser.add_argument('--no-skip-extracted-filings', action='store_false', dest='skip_extracted_filings',
                      help='Do not skip extracting from filings that have already been processed')
    parser.set_defaults(skip_extracted_filings=None)
    
    parser.add_argument('--include-signature', action='store_true', dest='include_signature',
                      help='Include signature sections in extracted filings')
    parser.add_argument('--no-include-signature', action='store_false', dest='include_signature',
                      help='Do not include signature sections in extracted filings')
    parser.set_defaults(include_signature=None)
    
    parser.add_argument('--remove-tables', action='store_true', dest='remove_tables',
                      help='Remove tables from extracted filings')
    parser.add_argument('--no-remove-tables', action='store_false', dest='remove_tables',
                      help='Do not remove tables from extracted filings')
    parser.set_defaults(remove_tables=False)
    
    # Execution flags
    parser.add_argument('--download', action='store_true',
                      help='Run the download_filings.py script after updating config')
    parser.add_argument('--extract', action='store_true',
                      help='Run the extract_items.py script after updating config')
    parser.add_argument('--run-all', action='store_true',
                      help='Run both download_filings.py and extract_items.py scripts')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set download and extract if run-all is specified
    if args.run_all:
        args.download = True
        args.extract = True
    
    return args

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Step 1: Fetch data from Wikipedia
    sp500_df = fetch_data()
    
    # If requested, print available sectors and industries and exit
    if args.list_sectors:
        print_available_sectors_and_industries(sp500_df)
        sys.exit(0)
    
    # Step 2: Filter companies based on selection criteria
    filtered_df = filter_companies(sp500_df, args)
    
    # Step 3: Save data to CSV
    csv_file = save_to_csv(filtered_df, args.csv)
    
    # Step 4: Update config.json with ticker symbols and parameters
    update_config(filtered_df['Symbol'], args.config, args)
    
    # Step 5: Run scripts if requested
    if args.download or args.extract:
        run_scripts(args)
    
    print("\nProcess completed successfully!")
    print(f"- {len(filtered_df)} S&P 500 companies data saved to {csv_file}")
    print(f"- {args.config} updated with selected S&P 500 ticker symbols and provided parameters")
    if args.download or args.extract:
        print("- Requested scripts have been executed")

if __name__ == "__main__":
    main()