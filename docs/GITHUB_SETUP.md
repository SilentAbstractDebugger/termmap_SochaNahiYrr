# Private GitHub repository: create, push, share

Run these commands from the repository root. Replace `<you>` and the judges' GitHub usernames.

```bash
# 0. safety first: nothing proprietary may be tracked
git init -b main                                  # skip if already a git repo
git add -A
bash scripts/check_no_data_tracked.sh             # must print OK; otherwise: git rm --cached <file>
git status --short | head -50                     # eyeball: no data/, resources/, models/, *.csv, *.xlsx, *.pkl, .env
git commit -m "TermMap: clinical term mapping engine (ICD-10-CM + SNOMED CT US)"

# 1. create a PRIVATE repo and push (GitHub CLI)
gh auth login                                     # once
gh repo create <you>/termmap --private --source=. --remote=origin --push
gh repo view <you>/termmap --json visibility      # must show "PRIVATE"

# 2. add judges as collaborators (read access)
gh api -X PUT repos/<you>/termmap/collaborators/<judge-username-1> -f permission=pull
gh api -X PUT repos/<you>/termmap/collaborators/<judge-username-2> -f permission=pull
gh api repos/<you>/termmap/invitations --jq '.[].invitee.login'   # pending invitations

# 3. optional: block accidental data commits locally
ln -sf ../../scripts/check_no_data_tracked.sh .git/hooks/pre-commit
```

Without the GitHub CLI: on github.com, open New repository, set Visibility to **Private**, and leave README, .gitignore
and licence unticked. Then:

```bash
git remote add origin https://github.com/<you>/termmap.git
git push -u origin main
```

Add collaborators under Settings -> Collaborators -> Add people.

## Check like a judge (incognito window)

1. Open an incognito / private window and go to `https://github.com/<you>/termmap`. You should get a **404**
   (proof the repo is private).
2. Sign in as a judge (or ask one to accept the invitation) and confirm the README renders: results block, links,
   no PENDING cells left after the final fill.
3. Browse `data/`, `resources/`, `models/`, `reports/private/`. They must not exist in the repo.
4. Search the repo for `.csv`, `.xlsx`, `.pkl`. There must be no results.
5. Open `docs/TermMap_writeup.pdf` in the browser. Check it is 2-3 pages and has no PENDING markers left.
