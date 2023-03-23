# import markdownify
import markdownify
from pathlib import Path
import re

ps = list(Path("Notion_DB/TPDM/").glob("**/**/*_*.htm"))
pattern = re.compile(r'\d+_\d+.htm')
for p in ps:
    if pattern.match(p.name):
        print(p.name)
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            html_text = f.read()

        # convert html to markdown
        h = markdownify.markdownify(html_text, heading_style="ATX")

        out = Path(p.parent,p.name.replace('htm','md'))
        # Write the Markdown to a file
        with open(out, 'w', encoding='utf-8') as f:
            f.write(h)
    
    
  
print(h)