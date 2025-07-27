# test_frontend.py
from pathlib import Path

current_dir = Path(__file__).parent
frontend_dir = current_dir / "frontend"
html_file = frontend_dir / "index.html"

print(f"Current directory: {current_dir}")
print(f"Frontend directory: {frontend_dir}")
print(f"HTML file: {html_file}")
print(f"Frontend dir exists: {frontend_dir.exists()}")
print(f"HTML file exists: {html_file.exists()}")

if html_file.exists():
    # Fixed: Added encoding='utf-8' and error handling
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            print(f"HTML file first 100 chars: {f.read()[:100]}")
    except UnicodeDecodeError:
        # Fallback: try with different encoding or error handling
        with open(html_file, 'r', encoding='utf-8', errors='replace') as f:
            print(f"HTML file first 100 chars (with replacements): {f.read()[:100]}")
    except Exception as e:
        print(f"Error reading HTML file: {e}")
else:
    print("❌ HTML file does not exist!")
