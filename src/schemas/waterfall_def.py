from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Tranche(BaseModel):
    """Represents a single debt or equity tranche in the CLO capital structure."""
    class_name: str = Field(
        description="Class identifier, e.g., 'Class A-1', 'Class B', 'Subordinated/Equity'"
    )
    target_rating: str = Field(
        description="Rating agency credit rating, e.g., 'AAA', 'AA', 'BBB', 'NR/Equity'"
    )
    principal_amount: float = Field(
        description="Initial par or principal balance in USD, e.g., 250000000.0"
    )
    coupon_type: Literal["floating", "fixed"] = Field(
        default="floating",
        description="Type of interest rate: floating (SOFR spread) or fixed"
    )
    spread_bps: float = Field(
        default=0.0,
        description="Spread over the benchmark rate in basis points (e.g., 135 for SOFR + 1.35%)"
    )
    fixed_rate: Optional[float] = Field(
        default=None,
        description="Fixed percentage coupon if not floating, e.g., 6.5 for 6.5%"
    )
    is_equity: bool = Field(
        default=False,
        description="True if this is the residual equity/subordinated tranche"
    )


class CoverageTest(BaseModel):
    """Overcollateralization (OC) or Interest Coverage (IC) test threshold."""
    test_type: Literal["OC", "IC"] = Field(
        description="Whether this is an Overcollateralization or Interest Coverage test"
    )
    applies_to_class: str = Field(
        description="The tranche or combined class tested, e.g., 'Class A/B' or 'Class C'"
    )
    trigger_ratio: float = Field(
        description="Minimum ratio threshold as a percentage, e.g., 121.5 for 121.5%"
    )
    cure_action: str = Field(
        default="Pay down senior principal until test is satisfied",
        description="Covenant remedy if test fails (typically diversion of cash to pay down senior notes)"
    )


class FeeStructure(BaseModel):
    """Senior fees, management fees, and administrative caps."""
    senior_admin_fee_cap: float = Field(
        default=200000.0,
        description="Annual dollar cap on trustee and administrative expenses"
    )
    senior_mgmt_fee_rate: float = Field(
        default=0.0015,
        description="Senior management fee rate as a decimal of collateral balance (e.g., 0.0015 = 15 bps)"
    )
    subordinated_mgmt_fee_rate: float = Field(
        default=0.0035,
        description="Subordinated management fee rate as a decimal (e.g., 0.0035 = 35 bps)"
    )
    incentive_fee_hurdle_irr: float = Field(
        default=0.12,
        description="Equity IRR hurdle before incentive fee applies (e.g., 0.12 = 12%)"
    )
    incentive_fee_share: float = Field(
        default=0.20,
        description="Manager share of residual cash flow above the hurdle (e.g., 0.20 = 20%)"
    )


class WaterfallStep(BaseModel):
    """A single sequential instruction in the Priority of Payments."""
    priority: int = Field(
        description="Execution order index (1, 2, 3, ...)"
    )
    payee: str = Field(
        description="Description or tranche recipient (e.g., 'Taxes and Trustee Fees', 'Class A Interest')"
    )
    payment_type: Literal["fees", "interest", "principal", "oc_cure", "residual_equity"] = Field(
        description="Classification of payment flow"
    )
    condition: Optional[str] = Field(
        default=None,
        description="Conditional check if any, e.g., 'Only if Class A/B OC test is breached'"
    )


class IndentureRules(BaseModel):
    """Complete parsed financial model extracted from the legal indenture."""
    deal_name: str = Field(
        default="Mock CLO Deal",
        description="Name of the CLO vehicle, e.g., 'Octagon Investment Partners XLV'"
    )
    total_target_par: float = Field(
        default=400000000.0,
        description="Total collateral asset pool target balance in USD (e.g., 400000000.0)"
    )
    tranches: List[Tranche] = Field(
        default_factory=list,
        description="Ordered list of all tranches issued from most senior to equity"
    )
    coverage_tests: List[CoverageTest] = Field(
        default_factory=list,
        description="All contractual OC and IC triggers"
    )
    fees: FeeStructure = Field(
        default_factory=FeeStructure,
        description="Fee caps and management payment rates"
    )
    ccc_bucket_limit: float = Field(
        default=0.075,
        description="Maximum portfolio allowance for CCC-rated loans before haircutting (default 7.5%)"
    )
    interest_waterfall: List[WaterfallStep] = Field(
        default_factory=list,
        description="Ordered steps for the Interest Proceeds Priority of Payments"
    )