#!/usr/bin/env python3
import argparse
import os
import re
import sys

def validate_name(name):
    if not name:
        return False, "Name cannot be empty."
    if len(name) > 64:
        return False, "Name must be 64 characters or less."
    if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name):
        return False, "Name must only contain unicode lowercase alphanumeric characters and hyphens, and must not start/end with a hyphen or contain consecutive hyphens."
    return True, ""

def validate_description(description):
    if not description:
        return False, "Description cannot be empty."
    if len(description) > 1024:
        return False, "Description must be 1024 characters or less."
    return True, ""

def main():
    parser = argparse.ArgumentParser(description="Create and validate a new Agent Skill in .agents/skills/")
    parser.add_argument("--name", required=True, help="Name of the skill (lowercase, alphanumeric, hyphens)")
    parser.add_argument("--description", required=True, help="Description of the skill (up to 1024 chars)")
    parser.add_argument("--license", default=None, help="Optional license (e.g. MIT, Apache-2.0)")
    parser.add_argument("--compatibility", default=None, help="Optional compatibility details")
    
    args = parser.parse_args()
    
    # Validation
    is_valid_name, err_name = validate_name(args.name)
    if not is_valid_name:
        print(f"Error: Invalid name '{args.name}'. {err_name}", file=sys.stderr)
        sys.exit(1)
        
    is_valid_desc, err_desc = validate_description(args.description)
    if not is_valid_desc:
        print(f"Error: Invalid description. {err_desc}", file=sys.stderr)
        sys.exit(1)
        
    # Resolve the directory of skills
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skills_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Fallback to current working directory checking if not structured this way
    if not (os.path.basename(skills_dir) == "skills" and os.path.basename(os.path.dirname(skills_dir)) == ".agents"):
        if os.path.isdir(os.path.join(os.getcwd(), ".agents", "skills")):
            skills_dir = os.path.join(os.getcwd(), ".agents", "skills")
        else:
            curr = os.path.abspath(os.path.dirname(__file__))
            found = False
            while curr != os.path.dirname(curr):
                potential = os.path.join(curr, ".agents", "skills")
                if os.path.isdir(potential):
                    skills_dir = potential
                    found = True
                    break
                curr = os.path.dirname(curr)
            if not found:
                print("Error: Could not locate '.agents/skills/' directory. Please run from within the repository.", file=sys.stderr)
                sys.exit(1)
                
    target_dir = os.path.join(skills_dir, args.name)
    if os.path.exists(target_dir):
        print(f"Error: Skill directory already exists at '{target_dir}'.", file=sys.stderr)
        sys.exit(1)
        
    # Create directories
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "references"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "assets"), exist_ok=True)
    
    # Build SKILL.md content
    frontmatter = [
        "---",
        f"name: {args.name}",
        f"description: {args.description}"
    ]
    if args.license:
        frontmatter.append(f"license: {args.license}")
    if args.compatibility:
        frontmatter.append(f"compatibility: {args.compatibility}")
    frontmatter.append("---")
    
    body = [
        "",
        f"# {args.name.replace('-', ' ').title()}",
        "",
        "## Overview",
        "",
        f"This skill provides instructions for: {args.description}",
        "",
        "## Workflow & Steps",
        "",
        "1. **Step One**: Detail what should be done first.",
        "2. **Step Two**: Detail the next step.",
        "",
        "## Gotchas",
        "",
        "- Gotcha 1: Detail a common non-obvious issue or configuration error.",
        "",
        "## Reference Materials",
        "",
        "See [the reference guide](references/REFERENCE.md) for deeper technical context.",
        ""
    ]
    
    skill_content = "\n".join(frontmatter + body)
    skill_file_path = os.path.join(target_dir, "SKILL.md")
    with open(skill_file_path, "w", encoding="utf-8") as f:
        f.write(skill_content)
        
    # Create a placeholder reference file
    ref_file_path = os.path.join(target_dir, "references", "REFERENCE.md")
    with open(ref_file_path, "w", encoding="utf-8") as f:
        f.write(f"# Reference for {args.name.replace('-', ' ').title()}\n\nAdd detailed technical references, APIs, or schemas here.\n")
        
    print(f"Successfully created skill '{args.name}' at '{target_dir}'.")
    print(f"Created files:\n  - {os.path.relpath(skill_file_path, os.getcwd())}\n  - {os.path.relpath(ref_file_path, os.getcwd())}")

if __name__ == "__main__":
    main()
