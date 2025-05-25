import os
import re

import argparse
from bs4 import BeautifulSoup
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def _html_tab_to_md(table: BeautifulSoup) -> str:
    """
    Convert an HTML table to Markdown format.
    
    Args:
        table (BeautifulSoup): The BeautifulSoup object representing the HTML table.
        
    Returns:
        str: The table in Markdown format.
    """
    markdown_table = []
    
    # Get all rows from the table
    rows = table.find_all('tr')
    if not rows:
        return ""
    
    # Process the header row (use the first row as header)
    header_cells = rows[0].find_all(['th'])
    if not header_cells:
        # If no th elements found, use td cells as header
        header_cells = rows[0].find_all('td')
        
    if not header_cells:
        # If still no cells found, return empty string
        return ""
        
    header = "| " + " | ".join([cell.get_text(strip=True) or " " for cell in header_cells]) + " |"
    markdown_table.append(header)
    
    # Add separator row
    separator = "| " + " | ".join(["---" for _ in range(len(header_cells))]) + " |"
    markdown_table.append(separator)
    
    # Process data rows (skip the first row if it was used as header)
    start_row = 1 if header_cells else 0
    for row in rows[start_row:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            # Handle case where data row might have different number of cells than header
            data_row = []
            for cell in cells:
                cell_text = cell.get_text(strip=True) or " "
                # Replace any pipe characters in the cell text to avoid breaking markdown table format
                cell_text = cell_text.replace("|", "\\|")
                data_row.append(cell_text)
            
            row_text = "| " + " | ".join(data_row) + " |"
            markdown_table.append(row_text)
    
    return "\n".join(markdown_table)


def convert_html_table_to_markdown(table) -> str:
    soup = BeautifulSoup(table, 'html.parser')
    table = soup.find('table')
    if table:
        return _html_tab_to_md(table)
    else:
        print(f"Row does not contain a valid table.")
        return None
    

def table_caption_detect(cur_table_chunk, row, idx, content) -> str:
    # Table caption is successfully recognized and included in 'table_caption'
    if len(row['table_caption']) > 0 and 'Table' in row['table_caption'][0]:
        cur_table_chunk = row['table_caption'][0] + '\n' + cur_table_chunk
        if len(row['table_footnote']) > 0:
            cur_table_chunk += '\n'.join(row['table_footnote'])
    
    # Table caption is not included in 'table_caption'
    else:
        prev_r_idx = idx - 1
        while True:
            r_prev = content.iloc[prev_r_idx]
            # Fail to find the table caption
            if r_prev['type'] != 'text' or (r_prev['type'] == 'text' and len(r_prev['text']) > 100):
                print(f"Row {prev_r_idx} does not contain a valid table caption.")
                break
            # Find the table caption
            elif r_prev['type'] == 'text':
                if 'Table' not in r_prev['text']:
                    prev_r_idx -= 1
                    continue
                else:
                    combined_caption = '\n'.join(content.iloc[prev_r_idx:idx]['text'].tolist())
                    cur_table_chunk = combined_caption + '\n' + cur_table_chunk
                    if len(row['table_footnote']) > 0:
                        cur_table_chunk += '\n'.join(row['table_footnote'])
                    break
    
    return cur_table_chunk


def chunk_text(text: str, tables: pd.DataFrame, metadata: dict):
    chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        chunk_size=600,
        chunk_overlap=100,
        length_function=lambda text: len(tokenizer.encode(text)),
        is_separator_regex=False,
    )
    
    chunk_index = 0
    chunk_texts = text_splitter.split_text(text)
    for chunk in chunk_texts:
        if chunk.strip():
            chunk_id = f"{metadata['file_name'].replace(' ', '_')}_chunk_{chunk_index}"
            chunk_data = {
                "chunk_id": chunk_id,
                "text": chunk.strip(),
                "metadata": metadata.copy()
            }
            chunks.append(chunk_data)
            chunk_index += 1
    
    # Add table data to chunks
    for index, row in tables.iterrows():
        if pd.isna(row['table_body_md']):
            print(f"Row {index} has no table body, skipping.")
            continue
        table_title = row['table_caption'][0] + '\n' if row['table_caption'] else ""
        table_text = row['table_body_md']
        table_metadata = metadata.copy()
        chunk_data = {
            "chunk_id": f"{metadata['file_name']}_chunk_{chunk_index}",
            "text": table_title + table_text,
            "metadata": table_metadata
        }
        chunks.append(chunk_data)
        chunk_index += 1
        
    return chunks            


def main():
    domain = 'economics'
    root_path = '/data/group_data/rag-robust-eval/data/economics/json_files'
    output_path = '/data/group_data/rag-robust-eval/data/economics/chunks'

    file_list = os.listdir(root_path)
    
    for file in file_list:
        file_name = os.path.join(root_path, file, 'auto', f'{file}_content_list.json')
        print('Loading ', file_name)
        content = pd.read_json(file_name)
        
        # Find the first row of type 'text' and contain 'OECD Economic Surveys'
        first_valid_row = content[(content['type'] == 'text') & (content['text'].str.contains('OECD Economic Surveys:', na=False))]
        if not first_valid_row.empty:
            idx_found = first_valid_row.index[0]
            metadata = content.iloc[idx_found]['text'].split(':')[1].strip() # country name year
            year = re.search(r'\d{4}', metadata).group(0)
            country = metadata.replace(year, '').strip()
            print("Found the metadata row at index: ", idx_found, "Country: ", country, " Year: ", year)
        else:
            print("!!! No row found containing 'OECD Economic Surveys.' for file: ", file_name)
            continue
        
        # metadata = content.iloc[0]['text']
        # if ':' in metadata:
        #     country = metadata.split(':')[1].strip().split(' ')[0]
        #     year = metadata.split(':')[1].strip().split(' ')[1]
        # else:
        #     country = metadata.replace('OECD Economic Surveys ', '').strip()
        #     # content.iloc[1]['text']: "JUNE 2022 "; use regex to extract the year
        #     year = re.search(r'\d{4}', content.iloc[1]['text']).group(0)
        metadata_dict = {
            'domain': domain,
            'file_name': file,
            'file_type': 'OECD Economic Survey',
            'file_country': country,
            'file_year': year,
            'chunk_type': None,   # text or table
            'chunk_page_idx': 0
        }
        
        # Locate the first table in the content
        table_idx = int(content[content['type'] == 'table'].index[0])
        # Filter catelog contents and extract only content after the first table
        content = content.iloc[table_idx - 1:].reset_index(drop=True).copy()
        
        chunks = []
        cur_text_chunk = ''
        for idx, row in content.iterrows():
            if row['type'] == 'text':
                if len((cur_text_chunk + row['text']).split()) < 600:
                    cur_text_chunk += row['text'] + '\n'
                else:
                    chunk_id = f"economics_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
                    metadata_dict['chunk_type'] = 'text'
                    metadata_dict['chunk_page_idx'] = row['page_idx']
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "text": cur_text_chunk.strip(),
                        "metadata": metadata_dict.copy()
                    }
                    chunks.append(chunk_data)
                    cur_text_chunk = row['text'] + '\n'
                    
            elif row['type'] == 'table':
                # Convert the table body to markdown
                if pd.isna(row['table_body']):
                    print(f"Row {idx} has no table body, skipping.")
                    continue
                cur_table_chunk = convert_html_table_to_markdown(row['table_body'])
                if cur_table_chunk is None:
                    print(f"Row {idx} has no valid table body, skipping.")
                    continue
                
                # Detect the table caption and combine it with the table body
                cur_table_chunk = table_caption_detect(cur_table_chunk, row, idx, content)
                
                chunk_id = f"economics_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
                metadata_dict['chunk_type'] = 'table'
                metadata_dict['chunk_page_idx'] = row['page_idx']
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": cur_table_chunk.strip(),
                    "metadata": metadata_dict.copy()
                }
                chunks.append(chunk_data)
            else:
                print(f'Row {idx} has a row type of {row["type"]}, skipping.')
                continue
        
        # Save the chunks to a new JSON file
        chunks_df = pd.DataFrame(chunks)
        output_file = os.path.join(output_path, f"{file}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        chunks_df.to_json(output_file, orient='records', lines=True)
        print(f"Processed {file} and saved to {output_file}")


if __name__ == "__main__":
    main()