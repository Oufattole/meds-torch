"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

api_nav = mkdocs_gen_files.Nav()
config_nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent

SRC_DIR = root / "src"


def process_file(path, is_yaml=False):
    if is_yaml:
        ref_root_dir = SRC_DIR / "meds_torch" / "configs"
    else:
        ref_root_dir = SRC_DIR  # / "meds_torch"

    module_path = path.relative_to(ref_root_dir).with_suffix("")
    doc_path = path.relative_to(ref_root_dir).with_suffix(".md")

    module_path = path.relative_to(ref_root_dir).with_suffix("")
    doc_path = path.relative_to(ref_root_dir).with_suffix(".md")
    full_doc_path = Path("reference") / ("config" if is_yaml else "api") / doc_path

    parts = tuple(module_path.parts)

    md_file_lines = []

    if not is_yaml:
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

            readme_path = Path("/".join(parts + ("README.md",)))
            if (ref_root_dir / readme_path).exists():
                md_file_lines.append(f'--8<-- "src/{str(readme_path)}"')
        elif parts[-1] == "__main__":
            return None

    if is_yaml:
        config_nav[parts] = doc_path.as_posix()
        md_file_lines.append(f"# {path.stem}")
        md_file_lines.append("```yaml")
        md_file_lines.append(path.read_text())
        md_file_lines.append("```")
    else:
        if parts:
            api_nav[parts] = doc_path.as_posix()
        ident = "meds_torch." + ".".join(parts) if ref_root_dir != SRC_DIR else ".".join(parts)
        if not ident:
            ident = path.stem
        md_file_lines.append(f"::: {ident}")

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write("\n".join(md_file_lines))

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

    return doc_path


# Process Python files
for path in sorted(SRC_DIR.rglob("*.py")):
    process_file(path)

# Process YAML files
for path in list(sorted(SRC_DIR.rglob("*.yaml")) + list(SRC_DIR.rglob("*.yml"))):
    process_file(path, is_yaml=True)

# Generate API reference navigation
with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.write("# API Reference\n\n")
    nav_file.writelines(api_nav.build_literate_nav())

# Generate Config reference navigation
with mkdocs_gen_files.open("reference/config/SUMMARY.md", "w") as nav_file:
    nav_file.write("# Config Reference\n\n")
    nav_file.writelines(config_nav.build_literate_nav())
