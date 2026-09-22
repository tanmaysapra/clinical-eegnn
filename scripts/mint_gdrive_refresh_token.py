#!/usr/bin/env python
"""One-time local script: does the interactive Google OAuth consent as YOUR
own Google account and prints a refresh token for `notebooks/src/
11_arch4r_fullrun.py`'s Drive checkpoint resume (`src/gdrive.py`).

Run this on your own machine (not Kaggle -- it opens a browser). See
`notebooks/README_ARCH4R_FULLRUN.md` for the one-time Google Cloud Console
setup (creating the OAuth client) this script assumes already happened.

    pip install google-auth-oauthlib
    python scripts/mint_gdrive_refresh_token.py --client-id ... --client-secret ...

Prints the refresh token once, to stdout only (never written to disk) --
paste it into the Kaggle Secret `GDRIVE_REFRESH_TOKEN` and it's done; this
script does not need to be run again unless you revoke access.
"""

import argparse

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/drive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", required=True)
    ap.add_argument("--client-secret", required=True)
    args = ap.parse_args()

    client_config = {
        "installed": {
            "client_id": args.client_id,
            "client_secret": args.client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)

    print("\nSave these three values as Kaggle Secrets on the notebook:\n")
    print(f"  GDRIVE_CLIENT_ID       = {args.client_id}")
    print(f"  GDRIVE_CLIENT_SECRET   = {args.client_secret}")
    print(f"  GDRIVE_REFRESH_TOKEN   = {creds.refresh_token}")


if __name__ == "__main__":
    main()
