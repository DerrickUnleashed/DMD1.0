import pandas as pd
import json  # To save data in a structured format

def extract_relevant_go_terms(csv_path, go_types, output_file="go_terms.txt"):
    """
    Extracts unique GO terms of specified types from a CSV file and saves them to a text file.

    Args:
        csv_path (str): Path to the CSV file.
        go_types (list): List of GO term types to extract.
        output_file (str): Path to the output text file.
    """
    df = pd.read_csv(csv_path)
    extracted_terms = {go_type: set() for go_type in go_types}

    for go_type in go_types:
        column_name = f'GO:{go_type}'
        if column_name in df.columns:
            for index, row in df.iterrows():
                terms_str = str(row[column_name])
                terms = [term.strip() for term in terms_str.split('|')]
                extracted_terms[go_type].update(terms)
        else:
            print(f"Warning: Column '{column_name}' not found in the CSV.")

    # Convert sets to lists for JSON serialization
    extracted_lists = {k: sorted(list(v)) for k, v in extracted_terms.items()}

    # Save to text file using JSON (for structure)
    with open(output_file, 'w') as f:
        json.dump(extracted_lists, f, indent=4)  # indent for readability

    print(f"Extracted GO terms saved to '{output_file}'")


if __name__ == "__main__":
    csv_file = 'Smaller_Dataset.csv'
    go_types_to_extract = ['Function', 'Process', 'Component']
    output_text_file = "go_terms.txt"
    extract_relevant_go_terms(csv_file, go_types_to_extract, output_text_file)