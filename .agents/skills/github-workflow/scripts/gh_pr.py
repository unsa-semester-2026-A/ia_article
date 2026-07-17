#!/usr/bin/env python3
"""
GitHub Pull Request Automation Script.
Creates a Pull Request using the GitHub CLI (gh).
"""

import argparse
import subprocess
import sys

def run_command(cmd):
    """Run a shell command and return its output or raise an error."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}", file=sys.stderr)
        print(f"Stdout: {result.stdout}", file=sys.stderr)
        print(f"Stderr: {result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()

def check_gh_auth():
    """Verify that gh CLI is logged in."""
    try:
        res = subprocess.run("gh auth status", shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            print("Warning: gh is not authenticated. Please run 'gh auth login' or ensure GH_TOKEN is set.", file=sys.stderr)
            return False
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Create a GitHub Pull Request using gh CLI.")
    parser.add_argument("--title", required=True, help="Title of the pull request")
    parser.add_argument("--body", required=True, help="Markdown body description of the pull request")
    parser.add_argument("--base", default="main", help="Base branch (default: main)")
    parser.add_argument("--draft", action="store_true", help="Create as a draft pull request")
    parser.add_argument("--web", action="store_true", help="Open the pull request in the web browser after creation")

    args = parser.parse_args()

    if not check_gh_auth():
        sys.exit(1)

    print("Checking current git remote and status...")
    # Get current branch
    current_branch = run_command("git branch --show-current")
    if current_branch == args.base:
        print(f"Error: You are currently on the base branch '{args.base}'. Cannot create a PR from the same branch.", file=sys.stderr)
        sys.exit(1)

    # Build gh command
    cmd = ["gh", "pr", "create"]
    cmd.append(f'--title "{args.title}"')
    cmd.append(f'--body "{args.body}"')
    cmd.append(f'--base "{args.base}"')

    if args.draft:
        cmd.append("--draft")
    if args.web:
        cmd.append("--web")

    cmd_str = " ".join(cmd)
    print(f"Running command: {cmd_str}")
    
    # Run creation
    output = run_command(cmd_str)
    print("\nPR Created Successfully!")
    print(output)

if __name__ == "__main__":
    main()
