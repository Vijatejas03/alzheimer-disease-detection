import re, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('reports/generate_ppt.py', encoding='utf-8') as f:
    content = f.read()

# Replace all problematic Unicode in print() calls only
def fix_print_line(m):
    s = m.group(0)
    s = s.replace('\u2705', '[OK]')
    s = s.replace('\u2192', '->')
    s = s.replace('\u26a0', '[!]')
    s = s.replace('\u2714', '[x]')
    s = s.replace('\u2716', '[x]')
    return s

new_content = re.sub(r'print\([^)]+\)', fix_print_line, content)

with open('reports/generate_ppt.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Patch applied.')
