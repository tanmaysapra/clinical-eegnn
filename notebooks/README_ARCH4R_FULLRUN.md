# Notebook 11 — manual setup (one time)

Kaggle notebook: **https://www.kaggle.com/code/tanmaysapra/architecture-4r-full-epoch-training-run**
(private -- ask Tanmay for access if you can't open it).

The notebook itself re-clones `tanmaysapra/clinical-eegnn` from GitHub on
every run (see its first cell) -- **it never reads local files**, so any
code change has to be pushed to GitHub before a run will pick it up. What
the automated push *can't* do is attach a dataset or add secrets to the
kernel (the Kaggle API silently drops both, or doesn't expose them at all) --
that's what this checklist is for. Do this once per fresh kernel; after
that, re-running the notebook resumes on its own.

## 1. Attach the dataset

Open the notebook link above → **Add Input** (right sidebar) → search
`akshaybe/updated-mimic-iv` → **Add**. The notebook falls back to
downloading it itself via `kagglehub` if you skip this, but that's slower
and less reliable -- attach it manually.

## 2. Add the GitHub token secret

You need a GitHub PAT with read access to `tanmaysapra/clinical-eegnn`.

Notebook editor → **Add-ons → Secrets** → **Add a new secret**:

| Label | Value |
|---|---|
| `GITHUB_TOKEN` | the PAT |

(If you already set this up for notebooks 09/10, it's the same secret --
nothing new here.)

## 3. Create the Google OAuth client (once, in Google Cloud Console)

The Drive checkpoints need to authenticate as a real Google account (not a
service account -- see `src/gdrive.py`'s docstring for why: a bare service
account has zero storage quota on a personal Gmail account and uploads fail).

1. https://console.cloud.google.com/ → create a project (any name).
2. **APIs & Services → Library** → enable the **Google Drive API**.
3. **APIs & Services → OAuth consent screen** → External → fill the
   required fields (app name, your email) → add your own Google account as
   a **test user** → save. (Test mode is fine -- this never needs
   Google's review.)
4. **APIs & Services → Credentials** → **Create Credentials → OAuth client
   ID** → application type **Desktop app** → create. Copy the **Client ID**
   and **Client secret**.

## 4. Mint a refresh token (once, on your own machine — not Kaggle)

```bash
pip install google-auth-oauthlib
python scripts/mint_gdrive_refresh_token.py --client-id <CLIENT_ID> --client-secret <CLIENT_SECRET>
```

This opens a browser, asks you to sign in and grant Drive access, then
prints three values. It does not write anything to disk.

## 5. Create a Drive folder for checkpoints

In Google Drive, make a folder (e.g. "arch4r-checkpoints"). Open it, copy
the id from the URL: `drive.google.com/drive/folders/<THIS_PART>`.

## 6. Add the four Drive secrets

Same **Add-ons → Secrets** panel as step 2:

| Label | Value |
|---|---|
| `GDRIVE_CLIENT_ID` | from step 3 |
| `GDRIVE_CLIENT_SECRET` | from step 3 |
| `GDRIVE_REFRESH_TOKEN` | from step 4's output |
| `GDRIVE_FOLDER_ID` | from step 5 |

## 7. Confirm GPU + internet are on

Notebook editor → right sidebar → **Accelerator: GPU T4 x1**, **Internet:
On**. Both should already be set (pushed that way), but double-check --
without internet the `git clone` in cell 1 fails immediately.

## 8. Trigger a run

**Save Version → Save & Run All (Commit)**. This has to be a manual click
in the UI every time -- the Kaggle MCP integration that pushes code changes
cannot attach secrets or the dataset to an automated run, so it only ever
updates the notebook's *content*, never triggers a real run.

## What happens after that

- Each run picks up wherever the last one left off: a fresh kernel with no
  prior Drive state draws a random seed and starts training
  `arch4r_encoder_only`; a kernel that finds `arch4r_fullrun_state.json` on
  Drive resumes the exact same seed/mode/epoch.
- Progress checkpoints to Drive every 5 epochs, so a session timing out or
  crashing loses at most ~25-30 minutes of work, not the whole run.
- One seed finishing all three modes (`encoder_only`, `fixed_k`,
  `adaptive_k`, 30 epochs each) plus the certified gate takes roughly
  5-6 hours total. If a single Kaggle session doesn't cover that, just
  repeat step 8 -- it'll keep resuming until `state["seed_complete"]` is
  `True`, at which point checkpoints + results move into a dated Drive
  subfolder and the next run starts a brand-new seed.
- The printed cell output at the end of each run shows this seed's
  Arch4R numbers next to the frozen phase-1 baseline table
  (`plain_xgboost`, `cascade_3stage`, `graph_free_encoder`,
  `arch2_diffusion`) -- that comparison table is also saved to
  `results/arch4r_fullrun_seed<seed>.json` as kernel output.
- To check status without opening the notebook: look at
  `arch4r_fullrun_state.json` in the Drive checkpoint folder --
  `current_mode` / `current_epoch` / `modes_done` say exactly where things
  stand.
