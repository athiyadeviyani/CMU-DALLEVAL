import json

# importing the module
import json

art_styles = set()
 
# Opening JSON file
with open('raw_dataset.json') as json_file:
    data = json.load(json_file)
 
    rows = data['rows']

    for row in rows:
        prompt_items = row['row']['Prompt'].split(', ')
        if len(prompt_items) > 5:
            for style in prompt_items[-5:]:
                art_styles.add(style)

with open('art_styles.txt', 'w') as file:
    for style in art_styles:
        if len(style) < 30:
            file.write(style.replace(',','').replace('.',''))
            file.write('\n')


