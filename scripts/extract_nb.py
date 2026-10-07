import json
import sys

def extract_code(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            code_cells.append(source)
            
    with open('extracted_hospital_xgboost.py', 'w', encoding='utf-8') as f:
        f.write("\n\n# --- CELL ---\n\n".join(code_cells))
        
    print("Extracted to extracted_hospital_xgboost.py")

extract_code('hospital_xgboost.ipynb')
