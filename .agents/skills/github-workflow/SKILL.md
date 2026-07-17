---
name: github-workflow
description: Guides the agent in executing Git and GitHub workflows (branching, conventional commits, staging, pushing, PR creation, merging, and syncing) and bundles automation scripts for commit creation and pull requests.
---

# GitHub Workflow Agent Skill

This skill provides guidelines and automation scripts for interacting with Git and GitHub. It helps AI agents follow standard branching strategies, conventional commit messages, and repository operations (using raw `git` and the GitHub CLI `gh`).

---

## 1. Branching Strategy

Follow a simplified feature-branch workflow:
- **Main Branch**: `main` or `master`. Never commit directly to the main branch unless it is a minor documentation fix or trivial configuration change.
- **Feature Branches**: Create descriptive branch names prefixed with `feat/`, `fix/`, `docs/`, or `refactor/` (e.g., `feat/add-yolo-inference`, `fix/broken-imports`).
- **Creation**:
  ```bash
  git checkout -b feat/my-feature-name
  ```

---

## 2. Commit Message Standards (Conventional Commits)

Commit messages MUST follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: A new feature.
- `fix`: A bug fix.
- `docs`: Documentation-only changes.
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.).
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `perf`: A code change that improves performance.
- `test`: Adding missing tests or correcting existing tests.
- `chore`: Changes to the build process or auxiliary tools and libraries (e.g., updating dependencies).

### Examples
- `feat(yolo): add support for YOLO26 segmentation masks`
- `fix(data): resolve file path issue in coco converter`
- `docs(readme): update setup instructions`

---

## 3. GitHub CLI (`gh`) Reference

Use the GitHub CLI (`gh`) for remote repository interactions. Ensure you check command options by running with `--help` before calling.

### Pull Requests
- **Create PR**:
  ```bash
  gh pr create --title "feat(yolo): add segmentation support" --body "Detailed description of changes..."
  ```
- **Check PR status**:
  ```bash
  gh pr status
  ```
- **View PR diff**:
  ```bash
  gh pr diff
  ```
- **Merge PR**:
  ```bash
  gh pr merge --merge --delete-branch
  ```

### Issues
- **List issues**:
  ```bash
  gh issue list
  ```
- **Create issue**:
  ```bash
  gh issue create --title "Bug: CUDA out of memory" --body "Steps to reproduce..."
  ```

### Repository Syncing
- **Sync fork with upstream**:
  ```bash
  gh repo sync
  ```

---

## 4. Bundled Automation Scripts

This skill bundles helper scripts in the `scripts/` directory to automate common Git operations:

### 1. Git Commit & Push Automation (`scripts/git_commit.py`)
A non-interactive Python script to stage files, validate the commit message against Conventional Commits, commit, and optionally push to the remote branch.
- **Usage**:
  ```bash
  python3 .agents/skills/github-workflow/scripts/git_commit.py --type feat --scope yolo --msg "add model inference script" --push
  ```

### 2. Create Pull Request Automation (`scripts/gh_pr.py`)
A non-interactive Python script that uses `gh` to create a pull request with predefined templates.
- **Usage**:
  ```bash
  python3 .agents/skills/github-workflow/scripts/gh_pr.py --title "feat(yolo): add inference script" --body "This PR adds the main YOLO26 inference script."
  ```

---

## 5. Workflow Execution Instructions

When performing a codebase change:
1. **Check Status**: Run `git status` to see modified files.
2. **Branch out**: Create a new branch: `git checkout -b <branch_name>`.
3. **Write Code**: Perform edits.
4. **Stage & Commit**: Use the `git_commit.py` script to stage files and create a Conventional Commit.
5. **Push & PR**: Push the branch and use the `gh_pr.py` script to open a PR for the user.
