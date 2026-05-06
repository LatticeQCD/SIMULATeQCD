# MEMO.md

# SIMULATeQCD MDWF-Clover Workflow

## Goal

Develop MDWF + Clover-Wilson support in SIMULATeQCD using:

- MacBook + Codex for editing
- cluster for compile/runtime
- private git synchronization
- NO GitHub

---

# Repositories

## MacBook repo

Purpose:
- Codex edits
- git commits
- documentation

Path example:

```bash
~/Projects/SimulateQCD-mdwf
```

Remotes:

```bash
git remote -v
```

Expected:

```text
cluster   otuslogin2:/pc2/users/e/ecfqcd01/git/SimulateQCD-mdwf.git
upstream  https://github.com/LatticeQCD/SIMULATeQCD.git
```

Disable accidental upstream push:

```bash
git remote set-url --push upstream DISABLED
```

---

## Cluster bare repo

Purpose:
- private synchronization hub

Path:

```bash
/pc2/users/e/ecfqcd01/git/SimulateQCD-mdwf.git
```

Never edit this directly.

---

## Cluster compile repo

Purpose:
- compile
- run
- debug

Path example:

```bash
~/work/SimulateQCD-mdwf-work
```

---

# Branch

Main branch:

```bash
mdwf-clover-kernel
```

---

# MacBook workflow

## Update branch

```bash
cd ~/Projects/SimulateQCD-mdwf

git checkout mdwf-clover-kernel

git pull cluster mdwf-clover-kernel
```

---

## Start Codex

```bash
codex
```

Use Codex ONLY on MacBook repo.

---

## Check changes

```bash
git status -sb

git diff
```

---

## Commit

```bash
git add .

git commit -m "message"
```

---

## Push to cluster

```bash
git push cluster mdwf-clover-kernel
```

---

# Cluster workflow

## Update cluster repo

```bash
cd ~/work/SimulateQCD-mdwf-work

git checkout mdwf-clover-kernel

git pull origin mdwf-clover-kernel
```

---

## Compile

```bash
cmake --build build -j 16
```

Save logs:

```bash
cmake --build build -j 16 2>&1 | tee build.log
```

---

# Direct cluster development

Allowed.

Useful for:
- CUDA fixes
- runtime debugging
- compiler issues

After editing on cluster:

```bash
git add .

git commit -m "fix: cluster issue"

git push origin mdwf-clover-kernel
```

Then sync MacBook:

```bash
cd ~/Projects/SimulateQCD-mdwf

git pull cluster mdwf-clover-kernel
```

---

# Merge upstream branch

Example:

```bash
cd ~/Projects/SimulateQCD-mdwf

git fetch upstream

git checkout mdwf-clover-kernel

git merge --no-ff upstream/Dslash_WilsonClover

git push cluster mdwf-clover-kernel
```

Then on cluster:

```bash
cd ~/work/SimulateQCD-mdwf-work

git pull origin mdwf-clover-kernel
```

---

# Rules

Before editing:

```bash
git status -sb
git pull
```

After editing:

```bash
git add .
git commit -m "message"
git push
```

Before switching machines:

```bash
git pull
```

Do NOT edit the same file independently on MacBook and cluster without syncing.

---

# Codex policy

Codex should help with:
- code navigation
- scaffolding
- documentation
- compile fixes
- small patches

Cluster is the source of truth for:
- compilation
- runtime correctness
- GPU behavior
- physics validation

---

# MDWF-Clover plan

1. inspect Dslash_WilsonClover
2. inspect MDWF code path
3. add kernel-selection scaffold
4. preserve old MDWF behavior
5. add clover-Wilson kernel path
6. validate c_sw = 0 limit
7. validate even-odd consistency
8. later: RHMC force support
