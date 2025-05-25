# Finance Data Collection

Note: code is based on [https://github.com/lefterisloukas/edgar-crawler](), for more detail, please visit their repo for more instruction

You can collect data by running one program `sec_edgar_script.py`

## Command-Line Arguments Overview

### CSV and Config Output

* `--csv`: *(str, optional)*

  Output CSV file path. Default is `nasdaq100_companies.csv` inside the dataset directory.
* `--config`: *(str, optional)*

  Output JSON config file path. Default is `config.json` inside the dataset directory.

### Company Selection Options

* `--sectors`: *(str list, optional)*

  Filter companies based on  **GICS Sectors** .

  Example:

  ```
  --sectors "Information Technology" "Health Care"
  ```
* `--industries`: *(str list, optional)*

  Filter companies based on  **GICS Sub-Industries** .

  Example:

  ```
  --industries "Semiconductors" "Software"
  ```
* `--symbol-regex`: *(str, optional)*

  Filter companies whose **stock ticker symbol** matches a regex pattern.

  Example:

  ```
  --symbol-regex "^A.*"
  ```
* `--name-regex`: *(str, optional)*

  Filter companies whose **company name** matches a regex pattern.

  Example:

  ```
  --name-regex "Tech|Software"
  ```
* `--limit`: *(int, optional)*

  Keep only the first N companies  **after filtering** .
* `--random`: *(int, optional)*

  Select N companies **randomly** after applying filters.
* `--list-sectors`: *(flag)*

  Print a list of all available GICS sectors and sub-industries found and exit.

---

### Year Settings

* `--auto-year`: *(flag)*

  Automatically set both **start_year** and **end_year** to the current year.
* `--start-year`: *(int, optional)*

  Set a custom **start year** for filings download.
* `--end-year`: *(int, optional)*

  Set a custom **end year** for filings download.

---

### Filing Parameters

* `--quarters`: *(int list, optional)*

  Specify which **quarters** (1–4) to download filings for.
* `--filing-types`: *(str list, optional)*

  Specify which **SEC filing types** to download (e.g., 10-K, 10-Q, 8-K).
* `--user-agent`: *(str, optional)*

  Specify a custom **User-Agent** string for SEC EDGAR API requests (must include your name/email per SEC rules).

---

### Folder and Metadata Settings

* `--raw-folder`: *(str, optional)*

  Folder path to store  **raw filings** .
* `--extracted-folder`: *(str, optional)*

  Folder path to store  **extracted filing content** .
* `--indices-folder`: *(str, optional)*

  Folder path to store  **EDGAR indices** .
* `--metadata-file`: *(str, optional)*

  Path to the **metadata CSV file** for filings.

---

### Boolean Flags

* `--skip-present-indices`: *(flag)*

  **Skip** downloading SEC indices if already present.
* `--no-skip-present-indices`: *(flag)*

  **Do not skip** downloading SEC indices even if present.
* `--skip-extracted-filings`: *(flag)*

  **Skip** extracting content from filings that have already been processed.
* `--no-skip-extracted-filings`: *(flag)*

  **Force re-extraction** from already processed filings.
* `--include-signature`: *(flag)*

  Include **signature sections** in extracted filing content.
* `--no-include-signature`: *(flag)*

  Exclude **signature sections** in extracted content.
* `--remove-tables`: *(flag)*

  Remove **tables** from the extracted filing content.
* `--no-remove-tables`: *(flag)*

  **Keep tables** in the extracted filing content.

---

### Script Execution

* `--download`: *(flag)*

  Run the `download_filings.py` script **after** setting up the config.
* `--extract`: *(flag)*

  Run the `extract_items.py` script **after** setting up the config.
* `--run-all`: *(flag)*

  Run **both** `download_filings.py` and `extract_items.py`.
