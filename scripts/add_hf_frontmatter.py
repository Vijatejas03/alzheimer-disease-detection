from pathlib import Path

p = Path("README.md")
content = p.read_text(encoding="utf-8")
frontmatter = """---
title: Alzheimer's Disease Detection & Explainability System
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.64.0"
app_file: app/app.py
pinned: false
---

"""

if not content.startswith("---"):
    p.write_text(frontmatter + content, encoding="utf-8")
    print("Frontmatter prepended successfully.")
else:
    print("Frontmatter already present.")
