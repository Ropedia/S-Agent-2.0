from pathlib import Path

from sagent.constants import SAGENT_REGISTRY_WEBSITE_URL

SUPABASE_URL = "https://ofhuhcpkvzjlejydnvyd.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"

CREDENTIALS_DIR = Path("~/.sagent").expanduser()
CREDENTIALS_PATH = CREDENTIALS_DIR / "credentials.json"
CALLBACK_PORT = 19284
HOSTED_CALLBACK_URL = f"{SAGENT_REGISTRY_WEBSITE_URL}/auth/cli-callback"
