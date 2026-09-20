# Branch Protection Setup (GitHub — human action required)

Repo-side git hooks (`.git/hooks/pre-push`) block direct pushes to `main`/`master` from this
machine, but hooks live locally and can be bypassed with `git push --no-verify` or from any
other clone. GitHub branch protection enforces the rule server-side and cannot be bypassed.
Both layers are needed.

## Steps

1. Go to the repository on GitHub → **Settings** → **Branches**.
2. Under "Branch protection rules", click **Add branch protection rule**.
3. Set **Branch name pattern** to `main`.
4. Enable:
   - **Require a pull request before merging**
   - **Require approvals**: at least 1
   - **Require status checks to pass before merging**
   - **Do not allow bypassing the above settings**
5. Under general repository settings, enable **Block force pushes** for `main`.
6. Save the rule.

## Why both layers matter

| Layer | Enforced by | Can be bypassed? |
|---|---|---|
| `.git/hooks/pre-push` | Local git client | Yes — `--no-verify`, or push from another clone |
| GitHub branch protection | GitHub server | No |

Local hooks catch accidental direct pushes early; GitHub branch protection is the actual
guarantee that `main` cannot be modified except through a reviewed pull request.
