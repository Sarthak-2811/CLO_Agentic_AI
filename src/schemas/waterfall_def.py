from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class Tranche(BaseModel):
    """Represents a single debt or equity tranche in the CLO capital structure."""
    class_name: str = Field(
        description="Class identifier, e.g., 'Class A-1', 'Class B', 'Subordinated/Equity'"
    )
    target_rating: str = Field(
        default="",
        description="Rating agency credit rating, e.g., 'AAA', 'AA', 'BBB', 'NR/Equity'"
    )
    principal_amount: Optional[float] = Field(
        default=None,
        description="Initial par or principal balance in USD, e.g., 250000000.0. Return null if not found."
    )
    coupon_type: Literal["floating", "fixed"] = Field(
        default="floating",
        description="Type of interest rate: floating (SOFR spread) or fixed"
    )
    spread_bps: Optional[float] = Field(
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
    trigger_ratio: Optional[float] = Field(
        default=None,
        description="Minimum ratio threshold as a percentage, e.g., 121.5 for 121.5%"
    )
    cure_action: str = Field(
        default="Pay down senior principal until test is satisfied",
        description="Covenant remedy if test fails (typically diversion of cash to pay down senior notes)"
    )


class FeeStructure(BaseModel):
    """Senior fees, management fees, and administrative caps."""
    senior_admin_fee_cap: Optional[float] = Field(
        default=200000.0,
        description="Annual dollar cap on trustee and administrative expenses. Return null if not found."
    )
    senior_mgmt_fee_rate: Optional[float] = Field(
        default=0.0015,
        description="Senior management fee rate as a decimal of collateral balance (e.g., 0.0015 = 15 bps). Return null if not found."
    )
    subordinated_mgmt_fee_rate: Optional[float] = Field(
        default=0.0035,
        description="Subordinated management fee rate as a decimal (e.g., 0.0035 = 35 bps). Return null if not found."
    )
    incentive_fee_hurdle_irr: Optional[float] = Field(
        default=0.12,
        description="Equity IRR hurdle before incentive fee applies (e.g., 0.12 = 12%). Return null if not found."
    )
    incentive_fee_share: Optional[float] = Field(
        default=0.20,
        description="Manager share of residual cash flow above the hurdle (e.g., 0.20 = 20%). Return null if not found."
    )

    @field_validator("senior_admin_fee_cap", "senior_mgmt_fee_rate",
                     "subordinated_mgmt_fee_rate", "incentive_fee_hurdle_irr",
                     "incentive_fee_share", mode="before")
    @classmethod
    def coerce_none_to_default(cls, v, info):
        """Allow LLM to return null — fall back to field default instead of crashing."""
        return v  # None is now acceptable; pydantic stores None as-is

    def effective_senior_admin_fee_cap(self) -> float:
        return self.senior_admin_fee_cap if self.senior_admin_fee_cap is not None else 200000.0

    def effective_senior_mgmt_fee_rate(self) -> float:
        return self.senior_mgmt_fee_rate if self.senior_mgmt_fee_rate is not None else 0.0015

    def effective_subordinated_mgmt_fee_rate(self) -> float:
        return self.subordinated_mgmt_fee_rate if self.subordinated_mgmt_fee_rate is not None else 0.0035

    def effective_incentive_fee_hurdle_irr(self) -> float:
        return self.incentive_fee_hurdle_irr if self.incentive_fee_hurdle_irr is not None else 0.12

    def effective_incentive_fee_share(self) -> float:
        return self.incentive_fee_share if self.incentive_fee_share is not None else 0.20


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
    total_target_par: Optional[float] = Field(
        default=400000000.0,
        description="Total collateral asset pool target balance in USD (e.g., 400000000.0). Return null if not found."
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
    ccc_bucket_limit: Optional[float] = Field(
        default=0.075,
        description="Maximum portfolio allowance for CCC-rated loans before haircutting (default 7.5%)"
    )
    interest_waterfall: List[WaterfallStep] = Field(
        default_factory=list,
        description="Ordered steps for the Interest Proceeds Priority of Payments"
    )

    def effective_total_par(self) -> float:
        """Return total_target_par, falling back to sum of tranche principals if null."""
        if self.total_target_par:
            return self.total_target_par
        principals = [t.principal_amount for t in self.tranches if t.principal_amount]
        return sum(principals) if principals else 400_000_000.0