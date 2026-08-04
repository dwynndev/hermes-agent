"""Money-math tests for usage_pricing (T13-F2: billing correctness).

Targets the coverage gaps measured at 63% (lines 974-1432): normalize_usage
anthropic/codex modes + reasoning-token extraction, resolve_billing_route
normalization, estimate_usage_cost status branches, and the compact
formatters. All pure/deterministic — the openrouter network path is
monkeypatched, everything else runs offline.

A cost-estimation bug is a user-visible money bug: overcounting cache tokens
as full-price input inflates spend; a wrong "unknown" status hides real cost.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from hypothesis import given, strategies as st

from agent.usage_pricing import (
    CanonicalUsage,
    PricingEntry,
    _normalize_anthropic_model_name,
    _normalize_bedrock_model_name,
    _to_decimal,
    _to_int,
    estimate_usage_cost,
    format_duration_compact,
    format_token_count_compact,
    normalize_usage,
    resolve_billing_route,
)


class TestNormalizeUsageModes:
    def test_anthropic_mode_maps_cache_fields(self):
        u = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=200,
        )
        c = normalize_usage(u, provider="anthropic")
        assert (c.input_tokens, c.output_tokens) == (100, 50)
        assert (c.cache_read_tokens, c.cache_write_tokens) == (1000, 200)

    def test_codex_mode_subtracts_cache_from_input_total(self):
        # API contract: input_tokens INCLUDES cached tokens; the canonical
        # input bucket must subtract them or cache reads bill twice.
        u = SimpleNamespace(
            input_tokens=1500,
            output_tokens=30,
            input_tokens_details=SimpleNamespace(
                cached_tokens=1000, cache_creation_tokens=200
            ),
        )
        c = normalize_usage(u, api_mode="codex_responses")
        assert c.input_tokens == 300
        assert c.cache_read_tokens == 1000
        assert c.cache_write_tokens == 200

    def test_codex_mode_clamps_negative_input_at_zero(self):
        # Malformed usage where details exceed the total must not produce
        # negative token counts downstream.
        u = SimpleNamespace(
            input_tokens=10,
            output_tokens=1,
            input_tokens_details=SimpleNamespace(cached_tokens=999),
        )
        c = normalize_usage(u, api_mode="codex_responses")
        assert c.input_tokens == 0

    def test_reasoning_tokens_from_completion_details(self):
        # Chat Completions shape (the bug this pins: reading only
        # output_tokens_details left reasoning invisible).
        u = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=500,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=21_000),
        )
        c = normalize_usage(u)
        assert c.reasoning_tokens == 21_000

    def test_reasoning_tokens_from_responses_details(self):
        u = SimpleNamespace(
            input_tokens=100,
            output_tokens=500,
            output_tokens_details=SimpleNamespace(reasoning_tokens=777),
        )
        c = normalize_usage(u, api_mode="codex_responses")
        assert c.reasoning_tokens == 777

    def test_empty_usage_returns_zero_canonical(self):
        c = normalize_usage(None)
        assert c.input_tokens == c.output_tokens == 0
        assert c.cache_read_tokens == c.cache_write_tokens == 0


class TestResolveBillingRoute:
    def test_openai_codex_is_subscription_included(self):
        r = resolve_billing_route("gpt-5", provider="openai-codex")
        assert r.billing_mode == "subscription_included"

    def test_slash_prefix_infers_known_providers_only(self):
        r = resolve_billing_route("anthropic/claude-opus-4-7")
        assert r.provider == "anthropic"
        assert r.model == "claude-opus-4-7"
        # unknown prefix is NOT split
        r2 = resolve_billing_route("somellm/whatever")
        assert r2.provider == "unknown"
        assert r2.model == "whatever"  # falls through to last-segment split

    def test_openai_api_slug_normalizes_to_openai(self):
        r = resolve_billing_route("gpt-5", provider="openai-api")
        assert r.provider == "openai"
        assert r.billing_mode == "official_docs_snapshot"

    def test_localhost_base_url_routes_to_custom_unknown(self):
        r = resolve_billing_route("mymodel", provider="", base_url="http://localhost:8080/v1")
        assert r.billing_mode == "unknown"

    def test_provider_case_and_whitespace_normalized(self):
        r = resolve_billing_route("  claude-opus-4-7 ", provider="  ANTHROPIC  ")
        assert r.provider == "anthropic"
        assert r.model == "claude-opus-4-7"


class TestBedrockNormalization:
    def test_cross_region_prefix_stripped(self):
        assert _normalize_bedrock_model_name("us.anthropic.claude-opus-4-7") == (
            "anthropic.claude-opus-4-7"
        )
        assert _normalize_bedrock_model_name("global.anthropic.claude-opus-4-7") == (
            "anthropic.claude-opus-4-7"
        )
        assert _normalize_bedrock_model_name("apac.anthropic.claude-opus-4-7") == (
            "anthropic.claude-opus-4-7"
        )
        assert _normalize_bedrock_model_name("au.anthropic.claude-sonnet-4-5") == (
            "anthropic.claude-sonnet-4-5"
        )

    def test_dotted_version_and_profile_suffixes_normalized(self):
        out = _normalize_bedrock_model_name(
            "au.anthropic.claude-sonnet-4.5-20250929-v1:0"
        )
        assert out == "anthropic.claude-sonnet-4-5"

    def test_anthropic_prefix_and_dot_versions(self):
        assert _normalize_anthropic_model_name("anthropic/Claude-Opus-4.7") == (
            "claude-opus-4-7"
        )


class TestEstimateUsageCostStatuses:
    def test_subscription_route_costs_zero_included(self):
        u = CanonicalUsage(input_tokens=10_000, output_tokens=5_000)
        r = estimate_usage_cost("gpt-5", u, provider="openai-codex")
        assert r.status == "included"
        assert r.amount_usd == Decimal(0)

    def test_no_pricing_entry_returns_unknown(self):
        u = CanonicalUsage(input_tokens=10)
        r = estimate_usage_cost("definitely-not-a-real-model-xyz", u)
        assert r.status == "unknown"
        assert r.amount_usd is None

    def test_missing_cache_read_price_returns_unknown_with_note(self):
        entry = PricingEntry(
            input_cost_per_million=Decimal("3"),
            output_cost_per_million=Decimal("15"),
            cache_read_cost_per_million=None,  # not published
            cache_write_cost_per_million=Decimal("3.75"),
            source="official_docs_snapshot",
        )
        u = CanonicalUsage(input_tokens=100, cache_read_tokens=5000)
        import agent.usage_pricing as up

        # Drive through the real pipeline with a stubbed lookup so the test
        # is deterministic regardless of snapshot contents.
        orig = up._lookup_official_docs_pricing
        up._lookup_official_docs_pricing = lambda route: entry
        try:
            r = estimate_usage_cost("claude-opus-4-7", u, provider="anthropic")
        finally:
            up._lookup_official_docs_pricing = orig
        assert r.status == "unknown"
        assert r.notes and "cache-read pricing unavailable" in r.notes[0]

    def test_openrouter_estimate_carries_reconciliation_note(self, monkeypatch):
        entry = PricingEntry(
            input_cost_per_million=Decimal("1"),
            output_cost_per_million=Decimal("2"),
            cache_read_cost_per_million=Decimal("0.1"),
            cache_write_cost_per_million=Decimal("1.25"),
            source="provider_models_api",
        )
        import agent.usage_pricing as up

        monkeypatch.setattr(up, "_openrouter_pricing_entry", lambda route: entry)
        u = CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        r = estimate_usage_cost("openai/gpt-5", u, provider="openrouter")
        # 1M input @ $1/M + 1M output @ $2/M = exactly $3
        assert r.amount_usd == Decimal("3")
        assert any("reconciled" in n for n in r.notes)


class TestCompactFormatters:
    def test_duration_boundaries(self):
        assert format_duration_compact(59) == "59s"
        assert format_duration_compact(60) == "1m"
        assert format_duration_compact(3599) == "60m"
        assert format_duration_compact(3600) == "1h"
        assert format_duration_compact(3660) == "1h 1m"
        assert format_duration_compact(86_400) == "1.0d"

    def test_token_count_units_and_trimming(self):
        assert format_token_count_compact(999) == "999"
        assert format_token_count_compact(1_500) == "1.5K"
        assert format_token_count_compact(1_234) == "1.23K"
        assert format_token_count_compact(12_345) == "12.3K"
        assert format_token_count_compact(123_456) == "123K"
        assert format_token_count_compact(2_500_000) == "2.5M"
        assert format_token_count_compact(-1_500) == "-1.5K"

    @given(st.integers(min_value=0, max_value=10**12))
    def test_token_count_never_empty_or_crashes(self, n):
        out = format_token_count_compact(n)
        assert out and out != "nan"
