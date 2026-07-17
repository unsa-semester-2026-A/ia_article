#!/usr/bin/env python3
"""
Conventional Commit and Push Automation Script.
Stages modified files, validates the message against the Conventional Commits format,
commits, and optionally pushes to origin.
"""

import argparse
import subprocess
import sys

VALID_TYPES = [
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "chore",
    "build", "ci", "revert"
]

def run_command(cmd):
    """Run a shell command and return its output or raise an error."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}", file=sys.stderr)
        print(f"Stdout: {result.stdout}", file=sys.stderr)
        print(f"Stderr: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()

def get_current_branch():
    """Get the name of the current active branch."""
    return run_command("git branch --show-current")

def main():
    parser = argparse.ArgumentParser(description="Automate staging, conventional committing, and pushing.")
    parser.add_argument("--type", required=True, help=f"Commit type (choose from: {', '.join(VALID_TYPES)})")
    parser.add_argument("--scope", required=True, help="Scope of the change (e.g., yolo, config, readme)")
    parser.add_argument("--msg", required=True, help="Commit message description")
    parser.add_argument("--files", nargs="*", default=[], help="Specific files to stage. If not specified, stages all modified and tracked files.")
    parser.add_argument("--push", action="store_true", help="Push to remote repository after committing")
    
    args = parser.parse_args()

    # Validate commit type
    if args.type not in VALID_TYPES:
        print(f"Error: Invalid commit type '{args.type}'. Must be one of: {', '.join(VALID_TYPES)}", file=sys.stderr)
        sys.exit(1)

    # Format commit message
    commit_msg = f"{args.type}({args.scope}): {args.msg}"
    print(f"Formatted Commit Message: '{commit_msg}'")

    # Stage files
    if args.files:
        files_str = " ".join(args.files)
        print(f"Staging specific files: {files_str}")
        run_command(f"git add {files_str}")
    else:
        # Default behavior: stage modified/tracked files (like git add -u) or git add .
        # Let's stage all modified, deleted, and untracked files by default using git add .
        print("Staging all changes...")
        run_command("git add .")

    # Check if there are changes to commit
    status = run_command("git status --porcelain")
    if not status:
        print("No changes staged to commit. Exiting.")
        sys.exit(0)

    # Commit
    print("Committing changes...")
    # Use raw string/escaping or list format for safer execution to avoid injection issues
    # Since run_command takes shell=True, we escape double quotes in the message
    escaped_msg = commit_msg.replace('"', '\\"')
    run_command(f'git commit -m "{escaped_msg}"')

    # Push to origin
    if args.push:
        branch = get_current_branch()
        if not branch:
            print("Error: Could not retrieve current branch name. Skip pushing.", file=sys.stderr)
            sys.exit(1)
        print(f"Pushing to remote origin branch '{branch}'...")
        # Automatically set upstream if branch doesn't exist on remote yet
        run_command(f"git push -u origin {branch}")
        print("Push complete.")
    else:
        print("Skipped push. Run 'git push' manually or use --push next time.")

if __name__ == "__main__":
    main()
