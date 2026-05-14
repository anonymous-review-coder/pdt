# Publishing Checklist

Use this directory as a fresh repository. Do not copy Git history from the
working research repository.

```bash
git init
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.com"
git add .
git commit -m "Initial anonymized review release"
git branch -M main
git remote add origin https://github.com/<anonymous-account>/<repository-name>.git
git push -u origin main
```

Before pushing, run:

```bash
rg -n "author|email|/Users/|<project-private-name>|<private-tool-name>" .
find . \( -name "__pycache__" -o -name "*.pyc" -o -name ".DS_Store" -o -name ".git" \) -print
```

The repository should contain source code, manifests, PDT initialization
matrices, and the bundled ETTh1 smoke-test data only.
