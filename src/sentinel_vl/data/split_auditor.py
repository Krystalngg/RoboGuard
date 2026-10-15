"""Split Auditor for enforcing video-level isolation and cross-protocol leakage detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
from sentinel_vl.data.schemas import SplitPartition


@dataclass
class AuditResult:
    """Detailed audit report for partition validation and contamination detection."""
    is_valid: bool
    internal_leaks: Dict[str, List[str]] = field(default_factory=dict)
    cross_protocol_leaks: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status_str = "PASSED (Zero Leakage)" if self.is_valid else "FAILED (LEAKAGE DETECTED)"
        lines = [f"Split Audit Result: {status_str}"]
        if self.internal_leaks:
            lines.append("  Internal Partition Overlaps:")
            for k, v in self.internal_leaks.items():
                lines.append(f"    - {k}: {v}")
        if self.cross_protocol_leaks:
            lines.append("  Cross-Protocol Contaminations:")
            for k, v in self.cross_protocol_leaks.items():
                lines.append(f"    - {k}: {v}")
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        return "\n".join(lines)


class SplitAuditor:
    """Audits dataset splits to ensure zero training-to-evaluation data contamination."""

    @staticmethod
    def audit_partition(partition: SplitPartition) -> AuditResult:
        """Audits single partition disjointness between train, validation, and test splits."""
        train_val = sorted(partition.train_video_ids & partition.val_video_ids)
        train_test = sorted(partition.train_video_ids & partition.test_video_ids)
        val_test = sorted(partition.val_video_ids & partition.test_video_ids)

        internal_leaks: Dict[str, List[str]] = {}
        if train_val:
            internal_leaks["train_vs_val"] = train_val
        if train_test:
            internal_leaks["train_vs_test"] = train_test
        if val_test:
            internal_leaks["val_vs_test"] = val_test

        errors: List[str] = []
        if internal_leaks:
            errors.append(f"Internal data leakage in partition '{partition.name}'")

        return AuditResult(
            is_valid=len(internal_leaks) == 0,
            internal_leaks=internal_leaks,
            errors=errors,
        )

    @staticmethod
    def audit_cross_protocol(p1: SplitPartition, p2: SplitPartition) -> AuditResult:
        """Checks for cross-protocol contamination (e.g. retrieval training videos evaluated in anomaly test)."""
        cross_leaks: Dict[str, List[str]] = {}

        # Check p1 train in p2 test
        p1_train_in_p2_test = sorted(p1.train_video_ids & p2.test_video_ids)
        if p1_train_in_p2_test:
            cross_leaks[f"{p1.name}_train_in_{p2.name}_test"] = p1_train_in_p2_test

        # Check p2 train in p1 test
        p2_train_in_p1_test = sorted(p2.train_video_ids & p1.test_video_ids)
        if p2_train_in_p1_test:
            cross_leaks[f"{p2.name}_train_in_{p1.name}_test"] = p2_train_in_p1_test

        # Check train in val across protocols
        p1_train_in_p2_val = sorted(p1.train_video_ids & p2.val_video_ids)
        if p1_train_in_p2_val:
            cross_leaks[f"{p1.name}_train_in_{p2.name}_val"] = p1_train_in_p2_val

        p2_train_in_p1_val = sorted(p2.train_video_ids & p1.val_video_ids)
        if p2_train_in_p1_val:
            cross_leaks[f"{p2.name}_train_in_{p1.name}_val"] = p2_train_in_p1_val

        errors: List[str] = []
        if cross_leaks:
            errors.append(f"Cross-protocol contamination detected between '{p1.name}' and '{p2.name}'")

        return AuditResult(
            is_valid=len(cross_leaks) == 0,
            cross_protocol_leaks=cross_leaks,
            errors=errors,
        )
