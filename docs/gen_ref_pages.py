"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

api_nav = mkdocs_gen_files.Nav()
config_nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src" / "meds_torch"


def process_python_files():
    for path in sorted(src.rglob("*.py")):
        # Skip the configs directory for API Reference
        if "configs" in path.parts:
            continue

        module_path = path.relative_to(src).with_suffix("")
        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("reference/api") / doc_path

        parts = tuple(module_path.parts)

        if parts[-1] == "__main__":
            continue

        md_file_lines = []

        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

            readme_path = path.parent / "README.md"
            if readme_path.exists():
                md_file_lines.append(f'--8<-- "{readme_path.relative_to(root)}"')

        if parts:  # Only add to navigation if parts is not empty
            api_nav[parts] = doc_path.as_posix()

        ident = "meds_torch"
        if parts:
            ident += "." + ".".join(parts)
        md_file_lines.append(f"::: {ident}")

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write("\n".join(md_file_lines))

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


def process_yaml_files():
    config_dir = src / "configs"
    for path in sorted(config_dir.rglob("*.yaml")) + sorted(config_dir.rglob("*.yml")):
        rel_path = path.relative_to(config_dir)
        doc_path = rel_path.with_suffix(".md")
        full_doc_path = Path("reference/config") / doc_path

        parts = tuple(rel_path.parts)

        config_nav[parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(f"# {path.stem}\n\n")
            fd.write("```yaml\n")
            fd.write(path.read_text())
            fd.write("\n```\n")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


process_python_files()
process_yaml_files()

with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.write("# API Reference\n\n")
    nav_file.writelines(api_nav.build_literate_nav())

with mkdocs_gen_files.open("reference/config/SUMMARY.md", "w") as nav_file:
    nav_file.write("# Config Reference\n\n")
    nav_file.writelines(config_nav.build_literate_nav())
