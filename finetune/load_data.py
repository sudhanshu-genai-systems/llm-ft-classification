"""
load_data.py
------------
Shared data-access layer for the BANKING77 dataset.

Reuses the exact data sources and canonical label order from the original
(now-unsupported) HuggingFace loading script `banking77.py`, so labels are
encoded to the same integer ids (0-76) that the old `PolyAI/banking77`
ClassLabel produced. That keeps your notebook results comparable with
papers/tutorials that used the original dataset.

Usage (notebook, project root on ``sys.path``):

    from finetune.load_data import load_banking77, LABELS, label2id, id2label
    train_df, test_df = load_banking77()

Usage (script from ``finetune/``):

    from load_data import load_banking77
    train_df, test_df = load_banking77()
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import urllib.request

# Same URLs as the original loading script
_BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
TRAIN_URL = f"{_BASE}/train.csv"
TEST_URL = f"{_BASE}/test.csv"

# Canonical label order copied verbatim from the original banking77.py
# (order matters: it defines the integer ids of the ClassLabel feature).
LABELS: list[str] = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

label2id: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
id2label: dict[int, str] = {i: name for i, name in enumerate(LABELS)}


def _read_csv(source: str, cache_dir: Path | None) -> pd.DataFrame:
    """Read a banking77 CSV from URL (with optional local caching) or path."""
    if source.startswith("http"):
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            local = cache_dir / source.rsplit("/", 1)[-1]
            if not local.exists():
                urllib.request.urlretrieve(source, local)
            return pd.read_csv(local)
        with urllib.request.urlopen(source) as resp:
            return pd.read_csv(io.BytesIO(resp.read()))
    return pd.read_csv(source)


def load_banking77(
    cache_dir: str | Path | None = "data/raw",
    encode_labels: bool = True,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load BANKING77 train/test as DataFrames with columns:
        text (str), category (str), and label (int, if encode_labels=True).

    cache_dir : local dir for caching the downloaded CSVs (None = no cache)
    """
    cache = Path(cache_dir) if cache_dir is not None else None
    train_df = _read_csv(TRAIN_URL, cache)
    test_df = _read_csv(TEST_URL, cache)

    for df in (train_df, test_df):
        df.columns = [c.strip() for c in df.columns]

    if encode_labels:
        for name, df in (("train", train_df), ("test", test_df)):
            unknown = set(df["category"]) - set(LABELS)
            if unknown:
                raise ValueError(f"Unknown categories in {name} split: {unknown}")
            df["label"] = df["category"].map(label2id.get).astype("int64")

    if validate:
        assert set(train_df["category"]) == set(LABELS), "train is missing some intents"
        assert set(test_df["category"]) <= set(LABELS)
        train_ok = bool(train_df["text"].notna().all())
        test_ok = bool(test_df["text"].notna().all())
        assert train_ok and test_ok

    return train_df, test_df


if __name__ == "__main__":
    tr, te = load_banking77()
    print(f"train: {len(tr):,} rows | test: {len(te):,} rows | intents: {tr['category'].nunique()}")
    print(tr.head())
