import os

from dotenv import load_dotenv

load_dotenv()

SAGENT_SUPABASE_URL = os.environ.get(
    "SAGENT_SUPABASE_URL", "https://hlqxxzsirfrgeqasvaps.supabase.co"
)
SAGENT_SUPABASE_PUBLISHABLE_KEY = os.environ.get(
    "SAGENT_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_RGMKhIM1NKzZ8bL3qCVNuA_5TsD6rbF"
)
