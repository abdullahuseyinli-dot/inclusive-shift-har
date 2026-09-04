# Active Semantic-Gauge Sentinel protocol

Status: locked before the first ASGS run, 2026-09-04.

ASGS is an explicitly labelled personalization study triggered by the failure of zero-shot posture resolution. It is not zero-shot and cannot be compared with a zero-shot score without showing its label cost and changed evaluation set.

For each new participant, the mobility class is fixed. The only permitted adaptation is retaining or swapping the sitting and standing output columns. Before reading a query label, ASGS ranks likely-stationary windows by the expected information between the normal and swapped semantic hypotheses: predicted stationary probability multiplied by absolute posture log odds. A queried label updates the log Bayes factor for swap versus identity. Querying stops when its magnitude reaches `log(3)` or the fixed budget is exhausted. An unresolved participant retains the original semantics. Every queried window is removed from evaluation.

Budgets of one, two, and three total queries per participant are all reported. Controls are the unchanged model on exactly the same remaining windows, a deterministic hash-random query strategy under the same candidate filter and Bayes rule, and the existing fixed SAR method using one labelled anchor from each posture class. The fixed SAR comparator needs two labels and uses class-aware anchor selection; this difference remains explicit.

Both frozen GSP and RMRP source OOF predictions are evaluated. No model is selected from the outcomes, and no target participant or DAGHAR result is accessed. Query decisions, observed query labels, accumulated evidence, removed-window masks, predictions, and reports are retained in the create-only audit package.

Any gain supports only the claim that sparse labelled interaction can resolve a participant-specific semantic ambiguity. Independent validation is required before suggesting deployment benefit.
