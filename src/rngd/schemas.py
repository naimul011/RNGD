"""Shared pydantic schemas — the contract every agent reads/writes against."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AgentTraceEntry(BaseModel):
    step: int
    agent: str
    action: str
    model: str
    status: Literal["ok", "degraded", "error"]
    latency_ms: float
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class CustomerProgram(BaseModel):
    unit_count_total: Optional[int] = None
    unit_mix_pct: dict[str, float] = Field(default_factory=dict)
    floor_count: Optional[int] = None
    amenities: list[str] = Field(default_factory=list)
    parking_preference: Optional[str] = None
    budget_usd: Optional[float] = None
    schedule_note: Optional[str] = None
    hard_requirements: list[str] = Field(default_factory=list)
    soft_requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class Easement(BaseModel):
    name: str
    side: str
    width_ft: float
    buildable: bool = False


class BuildableEnvelope(BaseModel):
    x_min_ft: float
    x_max_ft: float
    y_min_ft: float
    y_max_ft: float
    width_ft: float
    depth_ft: float
    area_sf: float


class ZoningRules(BaseModel):
    max_height_ft: Optional[float] = None
    max_stories: Optional[int] = None
    max_far: Optional[float] = None
    max_lot_coverage_pct: Optional[float] = None
    parking_ratio_per_unit: Optional[float] = None
    fire_lane_clear_width_ft: Optional[float] = None
    fire_lane_coverage_ft: Optional[float] = None
    ada_route_min_width_ft: Optional[float] = None
    max_exit_travel_ft: Optional[float] = None


class SiteConstraints(BaseModel):
    lot_frontage_ft: float
    lot_depth_ft: float
    lot_area_sf: float
    setbacks_ft: dict[str, float] = Field(default_factory=dict)
    easements: list[Easement] = Field(default_factory=list)
    buildable_envelope: Optional[BuildableEnvelope] = None
    zoning: ZoningRules = Field(default_factory=ZoningRules)
    citations: list[str] = Field(default_factory=list)
    cv_cross_check: dict[str, Any] = Field(default_factory=dict)


class BaaPPlan(BaseModel):
    module_counts: dict[str, int] = Field(default_factory=dict)
    total_unit_gsf: float = 0.0
    grossing_factor: float = 0.85
    estimated_building_gsf: float = 0.0
    estimated_footprint_sf_per_floor: float = 0.0
    core_count: int = 1
    utilization_notes: str = ""
    adjacency_plan: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class Conflict(BaseModel):
    id: str
    category: str
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    recommendation: str


class ConflictReport(BaseModel):
    conflicts: list[Conflict] = Field(default_factory=list)
    feasible_as_requested: bool = True
    program_adjustment_suggestions: list[str] = Field(default_factory=list)


class SimulationStepStats(BaseModel):
    iteration: int
    steps_run: int
    max_overlap_penalty: float
    converged: bool


class CVMetrics(BaseModel):
    """`geometry_passed` means only: no detected setback/easement/overlap
    violation. It says nothing about whether parking demand is met — that is
    `parking_shortfall` (required - detected, floored at 0), tracked
    separately because a capacity shortfall is a program/site conflict to
    resolve via conflict_agent, not a geometry defect the simulation can fix
    by nudging shapes around.
    """

    footprint_area_sf: float = 0.0
    lot_coverage_pct: float = 0.0
    setback_violations: list[str] = Field(default_factory=list)
    easement_violations: list[str] = Field(default_factory=list)
    overlap_violations: list[str] = Field(default_factory=list)
    parking_stalls_detected: int = 0
    parking_stalls_required: int = 0
    parking_shortfall: int = 0
    geometry_passed: bool = False


class CriticSuggestion(BaseModel):
    adjustments: list[str] = Field(default_factory=list)
    param_deltas: dict[str, float] = Field(default_factory=dict)
    continue_iterating: bool = False
    rationale: str = ""


class FinalDecision(BaseModel):
    decision: Literal[
        "APPROVE", "APPROVE_WITH_CONDITIONS", "REVISE", "INSUFFICIENT_INFORMATION"
    ] = "REVISE"
    confidence: float = 0.0
    narrative: str = ""
    unresolved_issues: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class ProjectState(BaseModel):
    project_id: str
    raw_rfp_text: str = ""
    raw_site_conditions: dict[str, Any] = Field(default_factory=dict)

    program: Optional[CustomerProgram] = None
    site: Optional[SiteConstraints] = None
    baap: Optional[BaaPPlan] = None
    conflicts: Optional[ConflictReport] = None

    simulation_history: list[SimulationStepStats] = Field(default_factory=list)
    cv_metrics_history: list[CVMetrics] = Field(default_factory=list)
    critic_history: list[CriticSuggestion] = Field(default_factory=list)

    site_plan_image_path: Optional[str] = None
    floor_plate_image_path: Optional[str] = None
    convergence_chart_path: Optional[str] = None

    final_decision: Optional[FinalDecision] = None
    trace: list[AgentTraceEntry] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
