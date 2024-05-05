input_file_path = "metadata.csv"
output_file_path = '/Users/patrick/Documents/speech_tech_final/no_punct.csv'

chars_to_remove = r".',\"(—!);-?:" # target characters to be removed

with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    
    for line in infile:
        for char in chars_to_remove:
            line = line.replace(char, '')

        outfile.write(line)

