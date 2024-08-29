"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"


def process_file(path, is_yaml=False):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api") / doc_path

    parts = tuple(module_path.parts)

    md_file_lines = []

    if not is_yaml:
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

            readme_path = Path("/".join(parts + ("README.md",)))
            if (src / readme_path).exists():
                md_file_lines.append(f'--8<-- "src/{str(readme_path)}"')
        elif parts[-1] == "__main__":
            return None

    nav[parts] = doc_path.as_posix()

    if is_yaml:
        md_file_lines.append(f"# {path.stem}")
        md_file_lines.append("```yaml")
        md_file_lines.append(path.read_text())
        md_file_lines.append("```")
    else:
        ident = ".".join(parts)
        md_file_lines.append(f"::: {ident}")

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write("\n".join(md_file_lines))

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    return doc_path


# Process Python files
for path in sorted(src.rglob("*.py")):
    process_file(path)

# Process YAML files
for path in list(sorted(src.rglob("*.yaml")) + list(src.rglob("*.yml"))):
    process_file(path, is_yaml=True)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
