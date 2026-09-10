"""Per-objective SSL loss components (P3.9 decomposition).

Each module holds the pure math for one pretraining objective so it can be unit-tested in
isolation. Stateful pieces (momentum encoder, contrastive queues, the stage-1 queue) remain
owned by ``SelfSupervisedModule`` for checkpoint stability; these functions receive the
tensors/modules they need and return losses + diagnostics.
"""
