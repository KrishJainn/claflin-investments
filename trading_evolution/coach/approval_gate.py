"""
Approval Gate for High-Risk Deployments.

Provides a boolean sign-off mechanism before enabling:
- Live capital deployment
- Aggressive strategy changes
- New experimental features

Human must explicitly approve by setting approved=True.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json


@dataclass
class ApprovalItem:
    """An item requiring approval."""
    item_id: str
    category: str  # "deployment", "strategy_change", "risk_increase"
    description: str
    
    # Risk assessment
    risk_level: str  # "low", "medium", "high", "critical"
    impact_description: str
    
    # Approval status
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    
    # Context
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "description": self.description,
            "risk_level": self.risk_level,
            "impact_description": self.impact_description,
            "approved": self.approved,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass 
class ApprovalGateState:
    """Current state of the approval gate."""
    
    # Global gate
    live_trading_approved: bool = False
    aggressive_mode_approved: bool = False
    
    # Pending approvals
    pending_items: List[ApprovalItem] = field(default_factory=list)
    
    # Approved items
    approved_items: List[ApprovalItem] = field(default_factory=list)
    
    # Rejected items
    rejected_items: List[ApprovalItem] = field(default_factory=list)
    
    # Metadata
    last_review: Optional[datetime] = None
    reviewer: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "live_trading_approved": self.live_trading_approved,
            "aggressive_mode_approved": self.aggressive_mode_approved,
            "pending_items": [i.to_dict() for i in self.pending_items],
            "approved_items": [i.to_dict() for i in self.approved_items],
            "rejected_items": [i.to_dict() for i in self.rejected_items],
            "last_review": self.last_review.isoformat() if self.last_review else None,
            "reviewer": self.reviewer,
        }


class ApprovalGate:
    """
    Approval gate for high-risk deployments.
    
    Usage:
        gate = ApprovalGate()
        
        # Check if live trading is approved
        if not gate.is_live_trading_approved():
            print("Live trading not approved")
            
        # Request approval for something
        gate.request_approval(item)
        
        # Human approves
        gate.approve(item_id, approved_by="krish")
    """
    
    def __init__(
        self,
        gate_file: str = "./approval_gate.json",
    ):
        """
        Initialize gate.
        
        Args:
            gate_file: Path to persistent gate state file
        """
        self.gate_file = Path(gate_file)
        self._state = ApprovalGateState()
        
        # Load existing state
        self._load_state()
    
    def _load_state(self):
        """Load state from file if exists."""
        if self.gate_file.exists():
            try:
                with open(self.gate_file, 'r') as f:
                    data = json.load(f)
                
                self._state.live_trading_approved = data.get("live_trading_approved", False)
                self._state.aggressive_mode_approved = data.get("aggressive_mode_approved", False)
                self._state.last_review = datetime.fromisoformat(data["last_review"]) if data.get("last_review") else None
                self._state.reviewer = data.get("reviewer")
                
                # Load items
                for item_data in data.get("pending_items", []):
                    self._state.pending_items.append(self._item_from_dict(item_data))
                for item_data in data.get("approved_items", []):
                    self._state.approved_items.append(self._item_from_dict(item_data))
                    
            except (json.JSONDecodeError, KeyError):
                pass
    
    def _save_state(self):
        """Save state to file."""
        with open(self.gate_file, 'w') as f:
            json.dump(self._state.to_dict(), f, indent=2)
    
    def _item_from_dict(self, data: Dict) -> ApprovalItem:
        """Create ApprovalItem from dict."""
        return ApprovalItem(
            item_id=data["item_id"],
            category=data["category"],
            description=data["description"],
            risk_level=data["risk_level"],
            impact_description=data["impact_description"],
            approved=data.get("approved", False),
            approved_by=data.get("approved_by"),
            approved_at=datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None,
            rejection_reason=data.get("rejection_reason"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
        )
    
    # =========================================================================
    # GLOBAL GATES
    # =========================================================================
    
    def is_live_trading_approved(self) -> bool:
        """Check if live trading is approved."""
        return self._state.live_trading_approved
    
    def is_aggressive_mode_approved(self) -> bool:
        """Check if aggressive mode is approved."""
        return self._state.aggressive_mode_approved
    
    def approve_live_trading(self, approved_by: str):
        """Approve live trading."""
        self._state.live_trading_approved = True
        self._state.last_review = datetime.now()
        self._state.reviewer = approved_by
        self._save_state()
    
    def revoke_live_trading(self, reason: str = ""):
        """Revoke live trading approval."""
        self._state.live_trading_approved = False
        self._save_state()
    
    def approve_aggressive_mode(self, approved_by: str):
        """Approve aggressive mode."""
        self._state.aggressive_mode_approved = True
        self._state.last_review = datetime.now()
        self._state.reviewer = approved_by
        self._save_state()
    
    def revoke_aggressive_mode(self, reason: str = ""):
        """Revoke aggressive mode approval."""
        self._state.aggressive_mode_approved = False
        self._save_state()
    
    # =========================================================================
    # ITEM APPROVALS
    # =========================================================================
    
    def request_approval(self, item: ApprovalItem):
        """Request approval for an item."""
        self._state.pending_items.append(item)
        self._save_state()
    
    def request_deployment_approval(
        self,
        item_id: str,
        description: str,
        risk_level: str = "medium",
        impact_description: str = "",
    ) -> ApprovalItem:
        """Request approval for a deployment."""
        item = ApprovalItem(
            item_id=item_id,
            category="deployment",
            description=description,
            risk_level=risk_level,
            impact_description=impact_description,
        )
        self.request_approval(item)
        return item
    
    def approve(
        self,
        item_id: str,
        approved_by: str,
    ) -> bool:
        """
        Approve a pending item.
        
        Args:
            item_id: ID of item to approve
            approved_by: Who approved
            
        Returns:
            True if item was found and approved
        """
        for i, item in enumerate(self._state.pending_items):
            if item.item_id == item_id:
                item.approved = True
                item.approved_by = approved_by
                item.approved_at = datetime.now()
                
                self._state.pending_items.pop(i)
                self._state.approved_items.append(item)
                self._state.last_review = datetime.now()
                self._state.reviewer = approved_by
                
                self._save_state()
                return True
        
        return False
    
    def reject(
        self,
        item_id: str,
        rejected_by: str,
        reason: str = "",
    ) -> bool:
        """Reject a pending item."""
        for i, item in enumerate(self._state.pending_items):
            if item.item_id == item_id:
                item.approved = False
                item.rejection_reason = reason
                
                self._state.pending_items.pop(i)
                self._state.rejected_items.append(item)
                
                self._save_state()
                return True
        
        return False
    
    def is_approved(self, item_id: str) -> bool:
        """Check if an item is approved."""
        return any(i.item_id == item_id for i in self._state.approved_items)
    
    def get_pending_items(self) -> List[ApprovalItem]:
        """Get all pending items."""
        return self._state.pending_items.copy()
    
    def get_status(self) -> Dict:
        """Get current gate status."""
        return {
            "live_trading_approved": self._state.live_trading_approved,
            "aggressive_mode_approved": self._state.aggressive_mode_approved,
            "pending_count": len(self._state.pending_items),
            "approved_count": len(self._state.approved_items),
            "rejected_count": len(self._state.rejected_items),
            "last_review": self._state.last_review.isoformat() if self._state.last_review else None,
            "reviewer": self._state.reviewer,
        }
    
    def generate_approval_file(self) -> str:
        """
        Generate a human-readable approval file for review.
        
        Returns:
            Path to generated file
        """
        lines = [
            "# APPROVAL GATE",
            "",
            f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
            "## Global Approvals",
            "",
            f"- Live Trading: {'✅ APPROVED' if self._state.live_trading_approved else '❌ NOT APPROVED'}",
            f"- Aggressive Mode: {'✅ APPROVED' if self._state.aggressive_mode_approved else '❌ NOT APPROVED'}",
            "",
        ]
        
        if self._state.pending_items:
            lines.extend([
                "---",
                "",
                "## Pending Approvals",
                "",
            ])
            
            for item in self._state.pending_items:
                lines.extend([
                    f"### {item.item_id}",
                    "",
                    f"- **Category**: {item.category}",
                    f"- **Risk Level**: {item.risk_level}",
                    f"- **Description**: {item.description}",
                    f"- **Impact**: {item.impact_description}",
                    f"- **Created**: {item.created_at.strftime('%Y-%m-%d')}",
                    "",
                    f"To approve: `gate.approve('{item.item_id}', 'your_name')`",
                    "",
                ])
        
        if self._state.last_review:
            lines.extend([
                "---",
                "",
                f"Last reviewed by **{self._state.reviewer}** on {self._state.last_review.strftime('%Y-%m-%d %H:%M')}",
            ])
        
        content = "\n".join(lines)
        
        approval_file = self.gate_file.parent / "APPROVAL_REQUIRED.md"
        with open(approval_file, 'w') as f:
            f.write(content)
        
        return str(approval_file)
