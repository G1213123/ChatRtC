import csv
import json

input_file = '<PATH_TO_CSV_FILE>'
output_file = '<PATH_TO_OUTPUT_FILE>'

with open(input_file, 'r') as f:
    reader = csv.reader(f)
    with open(output_file, 'w') as out:
        for row in reader:
            example = {'prompt': row[0], 'completion': row[1]}
            out.write(json.dumps(example) + '\n')