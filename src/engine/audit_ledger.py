"""
FestOS-Inspired Immutable Audit Ledger.
Maintains a tamper-evident, append-only log of every transformation,
cell edit, algorithm execution, and executive override.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import json
import pandas as pd


class AuditLedger:
    """
    Cryptographically chained audit ledger tracking all pipeline activities.
    """

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []
        self._prev_hash: str = "GENESIS_NODE_0000"

    def record_action(
        self,
        node_id: str,
        action_type: str,
        rationale: str,
        input_rows: int = 0,
        output_rows: int = 0,
        user_role: str = "Chief Analytics Officer",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Appends a new verified audit record with SHA-256 block hash chaining."""
        timestamp = datetime.now().isoformat()
        
        record_content = {
            "index": len(self._entries) + 1,
            "timestamp": timestamp,
            "node_id": node_id,
            "action_type": action_type,
            "rationale": rationale,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "rows_delta": output_rows - input_rows,
            "user_role": user_role,
            "metadata": metadata or {},
            "prev_hash": self._prev_hash,
        }

        # SHA-256 block hash
        raw_bytes = json.dumps(record_content, sort_keys=True).encode("utf-8")
        block_hash = hashlib.sha256(raw_bytes).hexdigest()
        record_content["hash"] = block_hash
        self._prev_hash = block_hash

        self._entries.append(record_content)
        return record_content

    def get_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(columns=["index", "timestamp", "node_id", "action_type", "rationale", "rows_delta", "hash"])
        df = pd.DataFrame(self._entries)
        return df[["index", "timestamp", "node_id", "action_type", "rationale", "input_rows", "output_rows", "rows_delta", "hash"]]

    def verify_integrity(self) -> bool:
        """Verifies hash-chain integrity of all audit entries."""
        curr_prev = "GENESIS_NODE_0000"
        for entry in self._entries:
            if entry["prev_hash"] != curr_prev:
                return False
            check_copy = dict(entry)
            expected_hash = check_copy.pop("hash")
            raw_bytes = json.dumps(check_copy, sort_keys=True).encode("utf-8")
            if hashlib.sha256(raw_bytes).hexdigest() != expected_hash:
                return False
            curr_prev = expected_hash
        return True
