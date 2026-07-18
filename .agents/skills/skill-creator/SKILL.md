---
name: skill-creator
description: Bootstrap, format, and validate custom Agent Skills following the Agent Skills specification. Use when building a new skill or editing an existing one in the repository.
license: MIT
---

# Skill Creator

This skill guides you (and other agents) in creating custom **Agent Skills** that adhere to the official format specifications and guidelines. It also bundles an automated creation/bootstrapping script.

---

## 1. Directory Structure

Every skill is a subdirectory under `.agents/skills/` containing at minimum a `SKILL.md` file:

```
.agents/skills/skill-name/
├── SKILL.md          # Required: metadata frontmatter + instructions
├── scripts/          # Optional: executable code (e.g. python, bash)
├── references/       # Optional: detailed technical reference md files
└── assets/           # Optional: templates, configurations, static data
```

---

## 2. Naming & Frontmatter Validation Rules

### Name Field
- Must be between 1 and 64 characters.
- Must match the parent directory name exactly (e.g., directory `pdf-processing/` must have `name: pdf-processing` in its `SKILL.md`).
- May only contain lowercase alphanumeric characters (`a-z`, `0-9`) and hyphens (`-`).
- Must not start or end with a hyphen (`-`).
- Must not contain consecutive hyphens (`--`).

### Description Field
- Must be between 1 and 1024 characters.
- Must describe what the skill does and when to use it.
- Should include specific keywords to help agents trigger it reliably.

### Optional Frontmatter Fields
- `license`: Short name or reference to a bundled license file.
- `compatibility`: Indicates environment requirements (e.g., `Requires Python 3.10+ and uv`, `Requires git`). Max 500 characters.
- `metadata`: Key-value mapping for custom attributes (e.g., version, author).
- `allowed-tools`: Space-separated string of pre-approved tools (experimental).

---

## 3. Creating a Skill with the Bundled Script

To automate the creation and validation of a new skill, run the Python utility script included with this skill:

```bash
python3 .agents/skills/skill-creator/scripts/create_skill.py --name <skill-name> --description "<description>"
```

### Script Arguments:
- `--name`: The name of the skill (validated against the specs).
- `--description`: The description of the skill (validated against the specs).
- `--license`: (Optional) The license type.
- `--compatibility`: (Optional) Compatibility requirement description.

This script:
1. Validates the name and description constraints.
2. Creates the target directory under `.agents/skills/<name>`.
3. Creates folders: `scripts/`, `references/`, `assets/`.
4. Creates a boilerplate `SKILL.md` with pre-filled frontmatter and standard sections.
5. Validates the generated skill files.

---

## 4. Best Practices for Content

- **Aim for Moderate Detail**: Keep `SKILL.md` under 500 lines. Place large technical references, schemas, and details in `references/REFERENCE.md` or separate files.
- **Progressive Disclosure**: Only load large files when needed (e.g., "Read `references/api-errors.md` if the API returns a non-200 status code").
- **Avoid Placeholders**: Provide concrete working code snippets rather than generic patterns or placeholders.
- **GOTCHAS Section**: Include a gotchas section in `SKILL.md` detailing non-obvious failure modes, environment quirks, or subtle bugs.
- **Non-Interactive Scripts**: Ensure all scripts run non-interactively (do not prompt the user for input). Use command-line arguments instead.
