# Relay 2026-08-14-2: INBOUND — sym/budget semantics answered (cooperative, bounded modulo poll gaps)

Axiom's answer to relay -1 ask 4, recorded verbatim in substance:
work_budget_scope installs a THREAD-LOCAL deadline; expensive sym
loops poll check_work_budget(); expiry throws work_expired, caught
at the installing boundary as a conservative rejection — never a
partial result. It bounds work inside a single rule application,
but only where polls exist (today: expand's distribute loops,
oracle/canonical recursions, including the (x^n)^(p/q) wedge that
historically polled nothing). Every rule fire and verify_edge runs
under an installed budget (3 s verify, 8 s successors/emit,
200-400 ms inverse). A hostile expression can overshoot
deadline_ms by at most one inter-poll stride — the guarantee is
per-poll-coverage, NOT a hard OS-level wall. Their ask back: an
input that hangs an unpolled path is a BUG they want reported,
not a doctrine violation.

HOUSE READING (policy, effective now): axiom in-process calls are
"bounded modulo poll gaps" — strictly stronger than sympy (which
is unbounded and SIGALRM-unsafe, the fork-only law's origin), but
not a hard wall. Split the fork law accordingly:
- Desk/interactive/gate use of the bridge: in-process, no fork
  wrapper needed; deadline_ms + expired is the contract.
- FARM LOOPS and any oracle-on-model-text at scale: keep the fork
  wall (belt over their suspenders) — the law's cost there is
  near zero and the blast radius of one unpolled-path hang in a
  10-hour farm is the whole farm.
- Any observed overshoot beyond ~one poll stride gets timed,
  minimized, and relayed as a bug report with the input.
