"""Audited third-party baseline provenance and local-reuse decisions.

This registry is descriptive.  Presence here does not mean that a model is
implemented, reproducible, protocol-compatible, or licensed for reuse.  In
particular, ``faithful_local_implementation`` must remain false until an
independent architecture and protocol validation has been completed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

ReuseDecision = Literal[
    "blocked_no_software_license",
    "blocked_no_official_code",
    "blocked_pending_protocol_safe_adapter",
    "eligible_for_clean_adapter",
    "paper_derived_loss_only",
]


@dataclass(frozen=True, slots=True)
class ThirdPartyBaselineSpec:
    """Immutable, claim-safe summary of one audited baseline source."""

    key: str
    display_name: str
    paper_url: str
    repository_url: str | None
    audited_commit: str | None
    software_license: str
    official_code_available: bool
    input_layout: str
    sampling_assumptions: tuple[str, ...]
    channel_assumptions: tuple[str, ...]
    reuse_decision: ReuseDecision
    faithful_local_implementation: bool
    decision_note: str


_SPECS = {
    "tinyhar": ThirdPartyBaselineSpec(
        key="tinyhar",
        display_name="TinyHAR",
        paper_url="https://doi.org/10.1145/3544794.3558467",
        repository_url="https://github.com/teco-kit/ISWC22-HAR",
        audited_commit="b84b89d09f6914fe93e82cde423e294042da2741",
        software_license="NOASSERTION (no repository software license found)",
        official_code_available=True,
        input_layout="official model: [batch, 1, time, channels]",
        sampling_assumptions=(
            "dataset-specific sampling rates and windows",
            "HAPT: 50 Hz, 2.56 seconds (128 samples)",
        ),
        channel_assumptions=(
            "dataset-specific multichannel input",
            "HAPT: 6 accelerometer/gyroscope channels",
            "original repository has no UCI-HAR dataset entry",
        ),
        reuse_decision="blocked_no_software_license",
        faithful_local_implementation=False,
        decision_note="Do not copy the official source or assign this name to a compact proxy.",
    ),
    "tinierhar": ThirdPartyBaselineSpec(
        key="tinierhar",
        display_name="TinierHAR",
        paper_url="https://doi.org/10.1145/3715071.3750410",
        repository_url="https://github.com/zhaxidele/TinierHAR",
        audited_commit="f2f1bbd7305689c374fb38cbb7ce853e7cefbb3f",
        software_license="NOASSERTION (no repository software license found)",
        official_code_available=True,
        input_layout="official model: [batch, 1, time, channels]",
        sampling_assumptions=(
            "paper standardizes all datasets to 4-second windows with 2-second overlap",
            "paper UCI-HAR setup therefore implies 200 samples at 50 Hz",
            "repository UCI-HAR configuration instead uses 2.56 seconds (128 samples)",
        ),
        channel_assumptions=("UCI-HAR repository configuration: 9 channels",),
        reuse_decision="blocked_no_software_license",
        faithful_local_implementation=False,
        decision_note=(
            "Do not copy the official source; the configured convolution-block count also needs "
            "resolution before a clean-room paper adaptation could be validated."
        ),
    ),
    "harmamba": ThirdPartyBaselineSpec(
        key="harmamba",
        display_name="HARMamba",
        paper_url="https://doi.org/10.1109/JIOT.2024.3463405",
        repository_url="https://github.com/dianoDouble/HARMamba",
        audited_commit="3afb0dced3c66702d698af354d7ec7e753e772ad",
        software_license="Apache-2.0",
        official_code_available=True,
        input_layout="official defaults are sequence-first internally; default seq_size=512, c_in=9",
        sampling_assumptions=(
            "paper UCI-HAR setup: 50 Hz, 128 samples, 50% overlap",
            "official model default: 512 samples, patch size 16",
        ),
        channel_assumptions=(
            "paper UCI-HAR setup: 9 channels",
            "official model default: 9 channels",
        ),
        reuse_decision="blocked_pending_protocol_safe_adapter",
        faithful_local_implementation=False,
        decision_note=(
            "Apache licensing permits reuse, but the audited entry point ignores dataset-specific "
            "arguments, pins nonportable local wheels, and selects checkpoints using test accuracy."
        ),
    ),
    "ccil": ThirdPartyBaselineSpec(
        key="ccil",
        display_name="CCIL",
        paper_url="https://doi.org/10.1609/aaai.v39i1.32077",
        repository_url=None,
        audited_commit=None,
        software_license="not applicable: no official software release located",
        official_code_available=False,
        input_layout="paper cross-dataset interface: [batch, 6, 1, 50]",
        sampling_assumptions=(
            "cross-dataset interface uses 50-sample windows",
            "cross-person interfaces use 125 or 200 samples depending on dataset",
        ),
        channel_assumptions=(
            "cross-dataset interface uses 6 inertial channels",
            "cross-person interfaces use 45, 27, or 6 channels depending on dataset",
        ),
        reuse_decision="paper_derived_loss_only",
        faithful_local_implementation=False,
        decision_note=(
            "Only equations 2-6 are implemented locally as a clearly marked paper-derived loss; "
            "the unreleased official training system cannot be claimed as reproduced."
        ),
    ),
    "bpd": ThirdPartyBaselineSpec(
        key="bpd",
        display_name="BPD",
        paper_url="https://doi.org/10.1145/3517252",
        repository_url="https://github.com/Jie-su/BPD",
        audited_commit="8b2338927c118d1daa5c602d48b6ae5156dd5966",
        software_license="Apache-2.0",
        official_code_available=True,
        input_layout="official backbones consume channel-first windows; effective window length 168",
        sampling_assumptions=(
            "paper: 168 samples and 50% overlap",
            "official Dataset default: 168 samples and stride 32",
            "dataset-specific rates span 25-100 Hz",
        ),
        channel_assumptions=(
            "dataset-specific inertial channels for PAMAP2, MHEALTH, DSADS, and GOTOV",
        ),
        reuse_decision="blocked_pending_protocol_safe_adapter",
        faithful_local_implementation=False,
        decision_note=(
            "Apache components may be adapted only after replacing boundary-unsafe windowing and "
            "target-every-epoch checkpoint selection; any such result is a protocol adaptation."
        ),
    ),
    "cmd_har": ThirdPartyBaselineSpec(
        key="cmd_har",
        display_name="CMD-HAR",
        paper_url="https://arxiv.org/abs/2503.21843",
        repository_url=None,
        audited_commit=None,
        software_license="not applicable: no official software release located",
        official_code_available=False,
        input_layout="paper UCI-HAR table: 128 time samples by 9 channels",
        sampling_assumptions=("paper UCI-HAR table: 50 Hz, 128 samples",),
        channel_assumptions=("paper UCI-HAR table: 9 channels",),
        reuse_decision="blocked_no_official_code",
        faithful_local_implementation=False,
        decision_note="Document as a disentanglement predecessor; do not invent an implementation.",
    ),
    "benchhar": ThirdPartyBaselineSpec(
        key="benchhar",
        display_name="BenchHAR / HAR-Bench",
        paper_url="https://arxiv.org/abs/2605.08296",
        repository_url="https://github.com/saiketa/HAR-Bench",
        audited_commit="358a377929b1b9c0a2cefc417c67f56d15d4d11c",
        software_license="NOASSERTION (no repository software license found)",
        official_code_available=True,
        input_layout="benchmark canonical tensor: [window, 120, 6] (or accelerometer-only 3 channels)",
        sampling_assumptions=("canonical benchmark interface: 20 Hz, 6 seconds, 120 samples",),
        channel_assumptions=("3-axis accelerometer or 6-axis accelerometer/gyroscope",),
        reuse_decision="blocked_no_software_license",
        faithful_local_implementation=False,
        decision_note=(
            "The benchmark adaptation cannot be copied without a software license and its 20 Hz/120 "
            "sample interface is not interchangeable with the locked 50 Hz/128 sample interface."
        ),
    ),
    "simmtm": ThirdPartyBaselineSpec(
        key="simmtm",
        display_name="SimMTM",
        paper_url="https://arxiv.org/abs/2302.00861",
        repository_url="https://github.com/thuml/SimMTM",
        audited_commit="169513bef74fb676e48d98a0e30f8823793f691c",
        software_license="NOASSERTION (no repository software license found)",
        official_code_available=True,
        input_layout="generic multivariate time-series input; HAR adapter is supplied by HAR-Bench",
        sampling_assumptions=(
            "no canonical smartphone-HAR sampling interface in upstream repository",
        ),
        channel_assumptions=("dataset-dependent; no upstream six-channel smartphone-HAR contract",),
        reuse_decision="blocked_no_software_license",
        faithful_local_implementation=False,
        decision_note=(
            "BenchHAR reports this as the strongest cross-subject SSL candidate, but neither the "
            "upstream repository nor the HAR-Bench adapter has a software license."
        ),
    ),
    "focal": ThirdPartyBaselineSpec(
        key="focal",
        display_name="FOCAL",
        paper_url="https://arxiv.org/abs/2310.20071",
        repository_url="https://github.com/tomoyoshki/focal",
        audited_commit="f6a989eef42e7590aacc0c025e5aed875c6025c9",
        software_license="MIT",
        official_code_available=True,
        input_layout="upstream implementation targets paired acoustic and seismic modalities",
        sampling_assumptions=("no official smartphone inertial sampling contract",),
        channel_assumptions=("no official accelerometer/gyroscope channel adapter",),
        reuse_decision="eligible_for_clean_adapter",
        faithful_local_implementation=False,
        decision_note=(
            "The upstream code is reusable under MIT, but a new inertial adapter and source-only "
            "validation are required; HAR-Bench's adapter itself is unlicensed."
        ),
    ),
    "liteway": ThirdPartyBaselineSpec(
        key="liteway",
        display_name="LITEWAY",
        paper_url="https://arxiv.org/abs/2608.09421",
        repository_url="https://github.com/dominique-nshimyimana/liteway",
        audited_commit="982100053db3a81b10a10d225711829473ac1f3d",
        software_license="NOASSERTION (no repository software license found)",
        official_code_available=True,
        input_layout="official model: [batch, 1, time, channels]",
        sampling_assumptions=(
            "dataset-specific windows with 50% overlap",
            "UCI-HAR: 50 Hz, 2.56 seconds (128 samples)",
        ),
        channel_assumptions=("UCI-HAR: 9 channels",),
        reuse_decision="blocked_no_software_license",
        faithful_local_implementation=False,
        decision_note=(
            "The 10 August 2026 paper is relevant lightweight related work, but its repository has "
            "no software license and its UCI interface is nine-channel rather than the benchmark's "
            "six-channel primary interface."
        ),
    ),
}

THIRD_PARTY_BASELINE_SPECS: Mapping[str, ThirdPartyBaselineSpec] = MappingProxyType(_SPECS)


def get_third_party_spec(key: str) -> ThirdPartyBaselineSpec:
    """Return a baseline spec, raising a precise error for unknown identifiers."""

    try:
        return THIRD_PARTY_BASELINE_SPECS[key]
    except KeyError as error:
        known = ", ".join(sorted(THIRD_PARTY_BASELINE_SPECS))
        raise KeyError(f"unknown third-party baseline {key!r}; known baselines: {known}") from error
