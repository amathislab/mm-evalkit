from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class GaitEvent:
    index: int
    event: str  # "heel_strike" or "toe_off"
    side: str  # "left" or "right"


@dataclass
class GaitPhase:
    start: int
    end: int
    phase: str


@dataclass
class ValidStride:
    start: int
    end: int
    side: str
    phases: list[str]


def get_feet_contacts(touch_sensors_data=None, grf=None, threshold=0):
    """
    Get feet contact information from touch sensor data.
    -1: none, 0: left, 1: right, 2: both
    """
    if touch_sensors_data is not None:
        feet_contacts = np.zeros(touch_sensors_data.shape[0])
        for i in range(touch_sensors_data.shape[0]):
            feet_contacts[i] = -1
            if (touch_sensors_data[i, 2] > threshold) or (touch_sensors_data[i, 3] > threshold):
                feet_contacts[i] = 0
                if (touch_sensors_data[i, 0] > threshold) or (touch_sensors_data[i, 1] > threshold):
                    feet_contacts[i] = 2
            elif (touch_sensors_data[i, 0] > threshold) or (touch_sensors_data[i, 1] > threshold):
                feet_contacts[i] = 1
    elif grf is not None:
        grf_right = grf["ground_force_right_vy"].values
        grf_left = grf["ground_force_left_vy"].values
        feet_contacts = np.zeros(grf_right.shape[0])
        for i in range(grf_right.shape[0]):
            feet_contacts[i] = -1
            if grf_left[i] > threshold:
                feet_contacts[i] = 0
                if grf_right[i] > threshold:
                    feet_contacts[i] = 2
            elif grf_right[i] > threshold:
                feet_contacts[i] = 1
    else:
        raise ValueError("Either touch_sensors_data or grf must be provided.")

    return feet_contacts


def detect_gait_events(feet_contacts) -> list[GaitEvent]:
    """
    Detect all gait events from contact labels.

    Contact encoding:
        -1 : no contacts
         0 : left only
         1 : right only
         2 : both contacts
    """

    events = []

    def has_left(x):
        return x in [0, 2]

    def has_right(x):
        return x in [1, 2]

    for i in range(1, len(feet_contacts)):
        prev = feet_contacts[i - 1]
        curr = feet_contacts[i]

        # -----------------------------------------
        # LEFT EVENTS
        # -----------------------------------------

        # Left heel-strike
        if not has_left(prev) and has_left(curr):
            events.append(GaitEvent(i, "heel_strike", "left"))

        # Left toe-off
        if has_left(prev) and not has_left(curr):
            events.append(GaitEvent(i, "toe_off", "left"))

        # -----------------------------------------
        # RIGHT EVENTS
        # -----------------------------------------

        # Right heel-strike
        if not has_right(prev) and has_right(curr):
            events.append(GaitEvent(i, "heel_strike", "right"))

        # Right toe-off
        if has_right(prev) and not has_right(curr):
            events.append(GaitEvent(i, "toe_off", "right"))

    return events


def extract_gait_phases(feet_contacts) -> list[GaitPhase]:
    """
    Extract gait phases frame-by-frame.

    Returns intervals labeled as:
        - double_support
        - left_stance
        - right_stance
        - flight
    """

    phases = []

    def label_state(x):

        if x == 2:
            return "double_support"

        elif x == 0:
            return "left_stance"

        elif x == 1:
            return "right_stance"

        elif x == -1:
            return "flight"

        else:
            raise ValueError(f"Invalid contact state: {x}")

    start = 0
    current_phase = label_state(feet_contacts[0])

    for i in range(1, len(feet_contacts)):
        phase = label_state(feet_contacts[i])

        if phase != current_phase:
            phases.append(
                GaitPhase(
                    start=start,
                    end=i - 1,
                    phase=current_phase,
                )
            )

            start = i
            current_phase = phase

    # close last segment
    phases.append(
        GaitPhase(
            start=start,
            end=len(feet_contacts) - 1,
            phase=current_phase,
        )
    )

    return phases


def extract_stride_intervals(events, side="right"):
    """
    Extract stride intervals:
    heel-strike -> next ipsilateral heel-strike
    """

    hs = [e.index for e in events if e.event == "heel_strike" and e.side == side]

    strides = []

    for i in range(len(hs) - 1):
        strides.append((hs[i], hs[i + 1]))

    return strides


def stride_phase_sequence(feet_contacts, start, end):
    """
    Compress contact labels into unique phase sequence.
    """

    def label(x):

        if x == 2:
            return "double_support"
        elif x == 0:
            return "left_stance"
        elif x == 1:
            return "right_stance"
        elif x == -1:
            return "flight"
        else:
            raise ValueError(f"Invalid contact state: {x}")

    seq = []

    prev = None

    for i in range(start, end + 1):
        phase = label(feet_contacts[i])

        if phase != prev:
            seq.append(phase)
            prev = phase

    return seq


def is_valid_stride(
    phases,
    side="right",
    allow_running=True,
):
    """
    Validate phase ordering inside a stride.
    """

    if side == "right":
        required = [
            "double_support",
            "left_stance",
        ]

    elif side == "left":
        required = [
            "double_support",
            "right_stance",
        ]

    else:
        raise ValueError("side must be 'right' or 'left'")

    # -----------------------------------------
    # Check required ordering
    # -----------------------------------------

    idx = 0

    for r in required:
        found = False

        while idx < len(phases):
            if phases[idx] == r:
                found = True
                idx += 1
                break

            idx += 1

        if not found:
            return False

    # -----------------------------------------
    # Running-specific flight phase
    # -----------------------------------------
    if not allow_running and "flight" in phases:
        return False

    return True


def extract_valid_strides(
    feet_contacts,
    events,
    side="right",
    min_duration=50,
    max_duration=150,
    allow_running=True,
):
    """
    Extract only biomechanically valid strides.
    """

    heel_strikes = [e.index for e in events if e.event == "heel_strike" and e.side == side]

    valid_strides = []

    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]

        duration = end - start

        # duration filtering
        if duration < min_duration:
            continue

        if max_duration is not None and duration > max_duration:
            continue

        phases = stride_phase_sequence(
            feet_contacts,
            start,
            end,
        )

        if is_valid_stride(
            phases,
            side=side,
            allow_running=allow_running,
        ):
            valid_strides.append(
                ValidStride(
                    start=start,
                    end=end,
                    side=side,
                    phases=phases,
                )
            )

    return valid_strides


def _side_contact_bool(feet_contacts, side="right"):
    """Boolean per-frame contact for one side (right: labels 1/2, left: labels 0/2)."""
    contacts = np.asarray(feet_contacts)
    if side == "right":
        return np.isin(contacts, [1, 2])
    elif side == "left":
        return np.isin(contacts, [0, 2])
    else:
        raise ValueError("side must be 'right' or 'left'")


def _debounce_bool(signal, min_contact=8, min_gap=8):
    """Clean a per-frame contact boolean by removing brief blips.

    A "fast undesired contact" (a heel-strike immediately followed by a toe-off) shows
    up as a very short run of True; a brief loss of contact mid-stance shows up as a
    very short run of False. Both spuriously split a single gait cycle into two.

    This first *closes* short interior gaps (False runs shorter than ``min_gap`` become
    True) and then *opens* short spikes (True runs shorter than ``min_contact`` become
    False), so only contact/flight phases that last long enough to be real survive. The
    very first and last frames are never fabricated into contact, so implicit
    start/end detection stays faithful to how the episode actually began/ended.
    """
    sig = np.asarray(signal, dtype=bool).copy()
    n = len(sig)
    if n == 0:
        return sig

    def runs(a):
        out = []
        i = 0
        while i < n:
            j = i
            while j < n and a[j] == a[i]:
                j += 1
            out.append((bool(a[i]), i, j))  # (value, start, end_exclusive)
            i = j
        return out

    # Close short interior gaps (do not touch runs that reach frame 0 or the last frame).
    for val, s, e in runs(sig):
        if (not val) and (e - s) < min_gap and s > 0 and e < n:
            sig[s:e] = True
    # Open short contact spikes.
    for val, s, e in runs(sig):
        if val and (e - s) < min_contact:
            sig[s:e] = False
    return sig


def extract_valid_strides_flexible(
    feet_contacts,
    side="right",
    target_cycle=100,
    min_cycle=60,
    max_cycle=150,
    debounce_contact=8,
    debounce_gap=8,
    min_swing=8,
    allow_implicit_start=True,
    allow_implicit_end=True,
    expected_n=None,
):
    """Extract gait cycles robustly, recovering cycles the strict extractor misses.

    A gait cycle normally runs from an ipsilateral (right) heel-strike to the next one:
    ``RHS -> LTO -> LHS -> RTO -> RHS``. The strict :func:`extract_valid_strides` needs
    two clean right heel-strikes bounding a cycle of 50-150 frames whose phases are
    exactly ``double_support -> left_stance``. Real rollouts of a short reference clip
    (~100 frames per cycle) routinely violate that in three benign ways, all of which
    this function tolerates:

    * **Missing initial heel-strike** — the episode starts with the right foot already
      down (double support / right stance), so the cycle in progress at frame 0 has no
      recorded RHS. Frame 0 is then treated as an implicit cycle start.
    * **Missing final heel-strike** — the episode ends mid-swing before the right foot
      lands again, but the cycle is almost complete. The last frame is treated as an
      implicit cycle end.
    * **Spurious split** — a brief undesired contact (a heel-strike immediately followed
      by a toe-off, or a brief loss of contact during stance) splits one cycle into two.
      The contact signal is debounced first so these blips disappear.

    Heel-strikes are re-detected on the *debounced* right-contact signal; frame 0 and/or
    the last frame are added as implicit cycle boundaries when the two conditions above
    hold; consecutive boundaries then become cycles, kept when their duration is in
    ``[min_cycle, max_cycle]`` (centred on ``target_cycle`` ~ 100) and the right foot
    actually swings (off for at least ``min_swing`` frames — rules out pure standing).
    If ``expected_n`` is given and more cycles survive than expected, the ones whose
    duration is closest to ``target_cycle`` are kept, so a rollout normally yields its
    expected number of gaits (``round(reference_length / 100)``).

    Returns a list of :class:`ValidStride`, drop-in compatible with
    :func:`extract_valid_strides`.
    """
    contacts = np.asarray(feet_contacts)
    n = len(contacts)
    if n < 2:
        return []

    right = _side_contact_bool(contacts, side=side)
    clean = _debounce_bool(right, min_contact=debounce_contact, min_gap=debounce_gap)

    # Heel-strikes = rising edges (off -> on) of the cleaned contact signal.
    heel_strikes = [i for i in range(1, n) if (not clean[i - 1]) and clean[i]]

    boundaries = list(heel_strikes)

    # Missing initial heel-strike: episode starts with the foot already in contact.
    if allow_implicit_start and clean[0]:
        boundaries = [0, *boundaries]

    # Missing final heel-strike: episode ends mid-swing with a near-complete trailing cycle.
    if allow_implicit_end and boundaries:
        last = boundaries[-1]
        trailing = (n - 1) - last
        if (not clean[-1]) and min_cycle <= trailing <= max_cycle:
            boundaries = [*boundaries, n - 1]

    boundaries = sorted(set(boundaries))

    candidates = []
    for start, end in pairwise(boundaries):
        duration = end - start
        if duration < min_cycle or duration > max_cycle:
            continue
        # The right foot must genuinely swing (leave the ground) within the cycle,
        # otherwise this is just a standing interval, not a stride.
        if int(np.sum(~clean[start:end])) < min_swing:
            continue
        candidates.append(
            ValidStride(
                start=start,
                end=end,
                side=side,
                phases=stride_phase_sequence(contacts, start, end),
            )
        )

    # If we over-segmented, keep the cycles closest to the target duration so the count
    # matches the expected number of gaits for the clip.
    if expected_n is not None and len(candidates) > expected_n:
        candidates = sorted(candidates, key=lambda s: abs((s.end - s.start) - target_cycle))[:expected_n]
        candidates = sorted(candidates, key=lambda s: s.start)

    return candidates


def get_gait_signal(signal, muscle_id, gait_indexes):
    signal_gait_cycles = []
    if len(gait_indexes) == 0:
        return signal_gait_cycles
    # signal_gait_cycles.append(signal[0:gait_indexes[0][1], muscle_id])
    # for c in range(1,len(gait_indexes)):
    #     signal_gait_cycles.append(signal[gait_indexes[c-1][1]:gait_indexes[c][1], muscle_id])
    for c in range(len(gait_indexes)):
        signal_gait_cycles.append(signal[gait_indexes[c][0] : gait_indexes[c][1], muscle_id])
    return signal_gait_cycles


def interpolate_gait_cycle(signal, muscle_id, gait_indexes):
    """
    Interpolate gait cycles to 100 samples.
    Also returns the original lengths of the gait cycles before interpolation.

    Returns:
    --------
    signal_gait_cycles_f_interp : list
        Interpolated gait cycles (each with 100 samples)
    gait_lengths : list
        Original lengths of each gait cycle before interpolation
    """
    signal_gait_cycles = get_gait_signal(signal, muscle_id, gait_indexes)

    signal_gait_cycle_lengths = [len(signal_gait_cycles[i]) for i in range(len(signal_gait_cycles))]

    signal_gait_cycles_f_interp = []
    for i in range(len(signal_gait_cycles)):
        f = interp1d(np.linspace(0, 1, len(signal_gait_cycles[i])), signal_gait_cycles[i])
        signal_gait_cycles_f_interp.append(f(np.linspace(0, 1, 100)))

    return signal_gait_cycles_f_interp, signal_gait_cycle_lengths
