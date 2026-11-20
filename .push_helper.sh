#!/bin/bash
set -e
USER="${1:?用法: ./push_helper.sh <GitHub用户名> [仓库名]}"
REPO="${2:-SA-AASC-EarlyWarning}"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$USER/$REPO.git"
sed -i '' "s|<your-account>|$USER|" CITATION.cff
git add CITATION.cff
git commit -m "Set repository URL in CITATION.cff" --amend --no-edit
git push -u origin main
