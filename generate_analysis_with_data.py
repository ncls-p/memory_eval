#!/usr/bin/env python3
"""
Generate an HTML analysis file with embedded benchmark data.
This avoids CORS issues when opening the file locally.
"""

import json

# Read the benchmark data
with open('benchmark_results.json', 'r') as f:
    benchmark_data = json.load(f)

# Read the HTML template
with open('analyse.html', 'r') as f:
    html_content = f.read()

# Insert the data as a JavaScript variable before the main script
data_script = f"""
    <script>
      // Embedded benchmark data
      const BENCHMARK_DATA = {json.dumps(benchmark_data)};
    </script>
"""

# Find where to insert the data (before the main script tag)
insert_position = html_content.find('    <script>\n      // Load actual data from benchmark_results.json')
if insert_position == -1:
    # Fallback: insert before </body>
    insert_position = html_content.find('</body>')

# Insert the data script
html_with_data = html_content[:insert_position] + data_script + '\n' + html_content[insert_position:]

# Write the new file
with open('analyse_with_data.html', 'w') as f:
    f.write(html_with_data)

print("Generated analyse_with_data.html with embedded benchmark data")
print(f"Total benchmark entries: {len(benchmark_data)}")
print(f"Unique QA pairs: {len(set(d['qa_pair_index'] for d in benchmark_data))}")