import os

from bs4 import BeautifulSoup
import pandas as pd

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
    

def main():
    domain = 'policy'
    root_path = '/data/group_data/rag-robust-eval/data/policy/json_files'
    output_path = '/data/group_data/rag-robust-eval/data/policy/chunks'
    file_list = os.listdir(root_path)
    
    # Load metadata list
    metadata_path = '/data/group_data/rag-robust-eval/data/policy/metadata.json'
    metadata_list = pd.read_json(metadata_path)
    metadata_list['id'] = metadata_list['id'].astype(str)
    
    for file in file_list:
        if os.path.exists(os.path.join(output_path, f"{file}.json")):
            print(f"File {file} already processed, skipping.")
            continue
        # Load the JSON file
        file_name = os.path.join(root_path, file, 'auto', f'{file}_content_list.json')
        print('Loading ', file_name)
        content = pd.read_json(file_name)
        
        # Find the metadata for the file
        metadata = metadata_list[metadata_list['id'] == file].iloc[0]
        metadata_dict = {
            'domain': domain,
            'file_name': metadata['id'],
            'file_type': metadata['planType'],
            'file_grantee': metadata['grantee']['granteeName'],
            'file_state': metadata['grantee']['state']['name'],
            'file_year': metadata['startYear'],
            'chunk_type': None,   # text or table
            'chunk_page_idx': 0
        }
        
        # Filter attachments
        # Find the first text beginning with 'Attachment'
        attachment_idx = content[content['type'] == 'text'].index[content.loc[content['type'] == 'text', 'text'].str.startswith('Attachment', na=False)].tolist()
        if attachment_idx:
            print(f"Found attachments at index {attachment_idx[0]}:", content.iloc[attachment_idx[0]]['text'])
            content = content.iloc[:attachment_idx[0]].reset_index(drop=True).copy()
        else:
            print("No attachments found, processing from the beginning.")
        
        chunks = []
        cur_text_chunk = ''
        for idx, row in content.iterrows():
            if row['type'] == 'text':
                if len((cur_text_chunk + row['text']).split()) < 600:
                    cur_text_chunk += row['text'] + '\n'
                else:
                    chunk_id = f"policy_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
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
                if len(row['table_caption']) > 0:
                    cur_table_chunk = '\n'.join(row['table_caption']) + '\n' + cur_table_chunk
                    if len(row['table_footnote']) > 0:
                        cur_table_chunk += '\n'.join(row['table_footnote'])
                
                chunk_id = f"policy_{metadata_dict['file_name'].replace(' ', '_')}_chunk_{len(chunks)}"
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
        chunks_df.to_json(output_file, orient='records')
        print(f"Processed {file} and saved to {output_file}")


if __name__ == "__main__":
    main()