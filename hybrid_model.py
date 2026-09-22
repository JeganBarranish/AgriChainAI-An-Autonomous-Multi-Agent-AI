"""
hybrid_model.py
----------------
AGF-BRC: Attention-Gated Fusion network with Boosted Residual Correction,
trained with a Profit-Weighted Cross-Entropy (PWCE) crop head so income is
baked into the objective, not just computed after the fact.

Paradigm map (for viva classification):
  - Stages 1-3: SUPERVISED LEARNING (neural network, gradient descent)
  - Stage 4:    SUPERVISED LEARNING (ensemble/stacking via XGBoost)
  - Stage 5:    RULE-BASED / DETERMINISTIC (implemented in farm_agent.py)
  - Feature encoding (one-hot, scaling): UNSUPERVISED PREPROCESSING only
    (no labels used; see farm_agent._encode_frame)

Novelty claim (precise):
  This is a COUPLING contribution, not a claim of inventing a new base
  algorithm. The novelty is that the same economic (profit) signal is
  injected at BOTH training time (PWCE reweights the crop head toward
  higher-profit classes) AND inference time (confidence-guarded profit
  re-ranking among agronomically plausible crops). Stages 1-4 are standard
  supervised components (dual-branch MLP, attention gate, multi-task heads,
  gradient-boosted residual stacking); Stage 5 is explicit deterministic
  decision logic. The hybrid value is the end-to-end pipeline that ties
  agronomic prediction to income-aware recommendation without letting price
  override low-confidence classifications.

STAGE 1 - Dual-Branch Encoder  [SUPERVISED - neural]
    Soil features (Soil_Type one-hot + Fertilizer_Used) and Climate features
    (Region one-hot + Rainfall_mm + Temperature_Celsius + Irrigation_Used +
    Weather_Condition one-hot) are encoded by two SEPARATE small MLPs,
    reflecting that soil chemistry/management and climate/water act through
    different pathways on crop suitability.

STAGE 2 - Learned Attention Gate  [SUPERVISED - neural]
    A lightweight gate looks at both branch embeddings together and learns a
    soft per-dimension weight before fusing them, instead of fixed
    concatenation. gate_statistics() reports the mean/std/spread of the gate
    activations so it can be verified the gate is not saturated at 0 or 1,
    and not identical for every sample.

STAGE 3 - Multi-task shared trunk, 3 heads  [SUPERVISED - neural, multi-task]
    - crop classification head, trained with PROFIT-WEIGHTED cross-entropy:
          L_crop = -sum_c  w_c * y_c * log(p_c)
      where w_c is each crop's normalized reference profit-per-hectare
      (see crop_metadata.get_profit_weight_vector). This repurposes
      CrossEntropyLoss's per-class `weight` argument -- normally used to
      correct class imbalance -- as an ECONOMIC weighting instead, so the
      classifier is nudged toward higher-income crops whenever multiple
      crops are agronomically plausible for the given conditions. This is
      the primary "novel" mechanism: income enters the model at training
      time, not only as a post-hoc filter.
      NOTE: the head emits RAW LOGITS. No softmax is applied before
      CrossEntropyLoss, which already includes log_softmax internally.
    - yield regression head, trained on STANDARDIZED yield
    - duration regression head, trained on duration / 100

    Both regression targets are rescaled to roughly unit variance so that
    neither MSE term can numerically dominate the classification loss.
    The yield standardizer is fitted on TRAINING DATA ONLY and passed in.

STAGE 4 - Gradient-Boosted Residual Corrector (hybrid stacking)  [SUPERVISED - ensemble]
    The neural net's penultimate embeddings + raw features are fed into
    XGBoost, trained to predict the RESIDUAL error of the yield/duration
    heads. At inference: final = nn_prediction + xgb_residual.
    This corrects ONLY the two regressors. Crop classification is never
    touched by Stage 4, so reported crop accuracy is purely the neural
    classifier's own performance.

STAGE 5 - Rule-based Economic Layer (crop_metadata.py + farm_agent.py)  [RULE-BASED]
    Deterministic decision logic, no learned parameters:
      1. Keep the top-4 crops by classification probability (of 6 total).
      2. Mark a candidate "plausible" if p >= max(0.03, top_p * 0.15).
      3. The argmax crop is costed with the model's own residual-corrected
         yield; the others use their reference-table yield, explicitly
         labeled, so the two estimate sources are never silently mixed.
      4. Best crop = argmax(profit) within the plausible subset only, so a
         low-confidence high-price crop can never win purely on price.
      5. Output contract is fixed: exactly 1 best crop + exactly 3
         alternatives (the rest of the top 4), ordered by probability.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBRegressor

# Reproducibility: multi-threaded float reduction is not associative, so
# results drift run-to-run unless intra-op parallelism is pinned. With the
# seeded DataLoader generator below this makes every run bit-identical.
torch.set_num_threads(1)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SoilBranch(nn.Module):
    """Stage 1 - soil/management pathway encoder [SUPERVISED]."""

    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ClimateBranch(nn.Module):
    """Stage 1 - climate/water pathway encoder [SUPERVISED]."""

    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Stage 2 - cross-branch attention fusion [SUPERVISED].

    A per-dimension sigmoid gate conditioned on both branch embeddings
    jointly, applied element-wise back to their concatenation so the network
    learns, per sample, how much to trust soil-side vs climate-side signals.
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, embed_dim), nn.Sigmoid(),
        )

    def forward(self, soil_emb, climate_emb):
        fused = torch.cat([soil_emb, climate_emb], dim=-1)
        attn_weights = self.gate(fused)
        return fused * attn_weights, attn_weights


class AGF_BRC_Net(nn.Module):
    """Stages 1-3 - Attention-Gated Fusion network [SUPERVISED, multi-task]."""

    def __init__(self, n_classes, soil_dim, climate_dim,
                 branch_hidden=64, branch_out=64, trunk_hidden=128, dropout=0.2):
        super().__init__()
        self.soil_branch = SoilBranch(soil_dim, hidden=branch_hidden, out_dim=branch_out)
        self.climate_branch = ClimateBranch(climate_dim, hidden=branch_hidden, out_dim=branch_out)
        self.attn_gate = AttentionGate(embed_dim=branch_out * 2)

        self.trunk = nn.Sequential(
            nn.Linear(branch_out * 2, trunk_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(), nn.Dropout(dropout),
        )

        # Raw logits -- CrossEntropyLoss applies log_softmax itself.
        self.crop_head = nn.Linear(trunk_hidden, n_classes)
        self.yield_head = nn.Linear(trunk_hidden, 1)
        self.duration_head = nn.Linear(trunk_hidden, 1)

    def forward(self, soil_x, climate_x):
        soil_emb = self.soil_branch(soil_x)
        climate_emb = self.climate_branch(climate_x)
        fused, attn_weights = self.attn_gate(soil_emb, climate_emb)
        trunk_out = self.trunk(fused)

        crop_logits = self.crop_head(trunk_out)
        yield_pred = self.yield_head(trunk_out).squeeze(-1)
        duration_pred = self.duration_head(trunk_out).squeeze(-1)
        return crop_logits, yield_pred, duration_pred, trunk_out, attn_weights


class HybridFarmModel:
    """
    Stages 1-4 wrapper: AGF-BRC neural net (PWCE crop loss) + XGBoost
    residual correctors [SUPERVISED stacking for yield/duration].
    Stage 5 (rule-based re-ranking) lives in farm_agent.recommend().
    """

    def __init__(self, n_classes, soil_dim, climate_dim, profit_weights=None,
                 branch_hidden=64, branch_out=64, trunk_hidden=128, dropout=0.2,
                 yield_mean=0.0, yield_std=1.0, device="cpu", seed=42):
        # Re-seed before weight init so each searched config starts from a
        # reproducible initialization regardless of search order.
        set_global_seed(seed)
        self.seed = seed
        self.device = device
        self.net = AGF_BRC_Net(
            n_classes, soil_dim, climate_dim,
            branch_hidden=branch_hidden, branch_out=branch_out,
            trunk_hidden=trunk_hidden, dropout=dropout,
        ).to(device)

        self.profit_weights = (
            torch.tensor(profit_weights, dtype=torch.float32).to(device)
            if profit_weights is not None else None
        )
        # Yield standardizer, fitted on TRAINING data only (passed in by caller).
        self.yield_mean = float(yield_mean)
        self.yield_std = float(yield_std) if yield_std else 1.0

        self.yield_corrector = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
        )
        self.duration_corrector = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
        )
        self._fitted_correctors = False
        self.history = []
        # Set by fit_neural. When a dataset has no yield/duration column these
        # stay False and predict() returns None for those outputs rather than
        # emitting untrained-head noise as if it were a prediction.
        self.trained_yield = False
        self.trained_duration = False

    @staticmethod
    def _masked_mse(pred, target, mask):
        """
        MSE over labelled entries only. `mask` is 1.0 where a target exists.
        Returns exactly 0 when nothing is labelled, so an absent task
        contributes no gradient at all (rather than pulling toward zeros).
        """
        n = mask.sum()
        if n.item() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return (((pred - target) ** 2) * mask).sum() / n

    # ---------- training ----------
    def fit_neural(self, soil_x, climate_x, crop_y, yield_y=None, duration_y=None,
                   val_data=None, epochs=150, lr=1e-3, batch_size=256,
                   patience=15, verbose=True, log_every=10):
        """
        Train Stages 1-3 with early stopping on VALIDATION crop accuracy.

        yield_y / duration_y may be None (dataset has no such column) or may
        contain NaN for individual unlabelled rows. Either way Stage 3 uses a
        MASKED multi-task loss: the regression terms are computed only over
        rows that actually carry a target, so no target is ever fabricated.
        The crop head always trains.

        val_data : (soil, climate, crop, yield, duration) tuple, where yield
                   and duration may be None. The held-out TEST set is never
                   passed here -- model selection uses validation only.
        """
        n = len(crop_y)
        t = lambda a, d: torch.tensor(a, dtype=d).to(self.device)
        soil_t, clim_t = t(soil_x, torch.float32), t(climate_x, torch.float32)
        crop_t = t(crop_y, torch.long)

        # Standardized yield + duration/100 keep both MSE terms near unit scale
        # so neither can swamp the classification loss.
        yield_arr, yield_mask = self._prepare_target(
            yield_y, n, lambda v: (v - self.yield_mean) / self.yield_std)
        dur_arr, dur_mask = self._prepare_target(
            duration_y, n, lambda v: v / 100.0)

        self.trained_yield = bool(yield_mask.any())
        self.trained_duration = bool(dur_mask.any())
        if verbose:
            print(f"    multi-task targets: crop=yes  "
                  f"yield={'yes' if self.trained_yield else 'MASKED OUT (absent)'}  "
                  f"duration={'yes' if self.trained_duration else 'MASKED OUT (absent)'}")

        yield_t, dur_t = t(yield_arr, torch.float32), t(dur_arr, torch.float32)
        ymask_t, dmask_t = t(yield_mask, torch.float32), t(dur_mask, torch.float32)

        ds = torch.utils.data.TensorDataset(
            soil_t, clim_t, crop_t, yield_t, dur_t, ymask_t, dmask_t)
        # Dedicated generator so batch order does not depend on how much
        # global RNG earlier configs in the search happened to consume.
        gen = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, generator=gen)

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        # PWCE: standard CrossEntropyLoss's per-class `weight` repurposed as
        # an economic (profit) weighting rather than a class-imbalance fix.
        ce_loss = nn.CrossEntropyLoss(weight=self.profit_weights)
        mse_loss = nn.MSELoss()

        best_val_acc, best_state, best_epoch, stale = -1.0, None, 0, 0
        self.history = []

        for epoch in range(1, epochs + 1):
            self.net.train()
            total_loss, correct, seen = 0.0, 0, 0
            for sb, cb, yb_crop, yb_yield, yb_dur, ym, dm in loader:
                opt.zero_grad()
                logits, yield_pred, dur_pred, _, _ = self.net(sb, cb)

                loss = (ce_loss(logits, yb_crop)
                        + 0.3 * self._masked_mse(yield_pred, yb_yield, ym)
                        + 0.3 * self._masked_mse(dur_pred, yb_dur, dm))
                loss.backward()
                opt.step()

                total_loss += loss.item() * len(sb)
                correct += (logits.argmax(1) == yb_crop).sum().item()
                seen += len(sb)

            train_loss, train_acc = total_loss / seen, correct / seen

            if val_data is not None:
                val_loss, val_acc = self._evaluate_loss(*val_data, ce_loss, mse_loss)
                self.history.append(dict(epoch=epoch, train_loss=train_loss,
                                         train_acc=train_acc, val_loss=val_loss,
                                         val_acc=val_acc))
                if val_acc > best_val_acc:
                    best_val_acc, best_epoch, stale = val_acc, epoch, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    stale += 1

                if verbose and (epoch % log_every == 0 or epoch == 1):
                    print(f"    epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} "
                          f"acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

                if stale >= patience:
                    if verbose:
                        print(f"    early stop at epoch {epoch} "
                              f"(no val improvement for {patience} epochs)")
                    break
            elif verbose and epoch % log_every == 0:
                print(f"    epoch {epoch:3d}/{epochs} | train loss {train_loss:.4f} "
                      f"acc {train_acc:.4f}")

        # Restore the checkpoint that scored best on VALIDATION.
        if best_state is not None:
            self.net.load_state_dict(best_state)
            if verbose:
                print(f"    restored best checkpoint from epoch {best_epoch} "
                      f"(val acc {best_val_acc:.4f})")
        return best_val_acc

    @staticmethod
    def _prepare_target(values, n, transform):
        """
        Returns (scaled_values, mask). `values` may be None (task absent) or
        contain NaN for individual unlabelled rows. Unlabelled positions are
        filled with 0.0 purely as a placeholder; their mask entry is 0 so they
        never contribute to the loss.
        """
        if values is None:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(arr).astype(np.float32)
        scaled = np.where(mask > 0, transform(np.nan_to_num(arr, nan=0.0)), 0.0)
        return scaled.astype(np.float32), mask

    def _evaluate_loss(self, soil_x, climate_x, crop_y, yield_y, duration_y,
                       ce_loss, mse_loss):
        self.net.eval()
        with torch.no_grad():
            t = lambda a, d: torch.tensor(a, dtype=d).to(self.device)
            logits, yp, dp, _, _ = self.net(
                t(soil_x, torch.float32), t(climate_x, torch.float32))
            crop_t = t(crop_y, torch.long)

            n = len(crop_y)
            y_arr, y_mask = self._prepare_target(
                yield_y, n, lambda v: (v - self.yield_mean) / self.yield_std)
            d_arr, d_mask = self._prepare_target(duration_y, n, lambda v: v / 100.0)

            loss = (ce_loss(logits, crop_t)
                    + 0.3 * self._masked_mse(yp, t(y_arr, torch.float32),
                                             t(y_mask, torch.float32))
                    + 0.3 * self._masked_mse(dp, t(d_arr, torch.float32),
                                             t(d_mask, torch.float32)))
            acc = (logits.argmax(1) == crop_t).float().mean().item()
        return loss.item(), acc

    def gate_statistics(self, soil_x, climate_x) -> dict:
        """
        Stage 2 sanity check: confirm the attention gate is actually doing
        something -- not saturated near 0/1 and not identical per sample.
        """
        self.net.eval()
        with torch.no_grad():
            _, _, _, _, gates = self.net(
                torch.tensor(soil_x, dtype=torch.float32).to(self.device),
                torch.tensor(climate_x, dtype=torch.float32).to(self.device))
        g = gates.cpu().numpy()
        return {
            "mean": float(g.mean()),
            "std_within_sample": float(g.std(axis=1).mean()),
            "std_across_samples": float(g.mean(axis=1).std()),
            "frac_saturated_low": float((g < 0.05).mean()),
            "frac_saturated_high": float((g > 0.95).mean()),
        }

    def _forward_numpy(self, soil_x, climate_x):
        self.net.eval()
        with torch.no_grad():
            logits, yield_pred, dur_pred, trunk_out, _ = self.net(
                torch.tensor(soil_x, dtype=torch.float32).to(self.device),
                torch.tensor(climate_x, dtype=torch.float32).to(self.device))
        return (
            logits.cpu().numpy(),
            yield_pred.cpu().numpy() * self.yield_std + self.yield_mean,
            dur_pred.cpu().numpy() * 100.0,
            trunk_out.cpu().numpy(),
        )

    def fit_residual_correctors(self, soil_x, climate_x, yield_y, duration_y):
        """
        Stacking: XGBoost learns the NN's residual error on its own
        (already-trained) predictions, using the NN's trunk embeddings + raw
        features as its input. Crop classification is NOT affected.

        Fits only the correctors whose task actually has targets. If the
        dataset supplies neither yield nor duration, Stage 4 is a no-op and
        reports so -- there is no residual to correct.
        """
        _, nn_yield, nn_dur, embeddings = self._forward_numpy(soil_x, climate_x)
        X_stack = np.concatenate([embeddings, soil_x, climate_x], axis=1)

        fitted = []
        if self.trained_yield and yield_y is not None:
            m = np.isfinite(np.asarray(yield_y, dtype=np.float64))
            if m.any():
                self.yield_corrector.fit(X_stack[m], np.asarray(yield_y)[m] - nn_yield[m])
                fitted.append("yield")
        if self.trained_duration and duration_y is not None:
            m = np.isfinite(np.asarray(duration_y, dtype=np.float64))
            if m.any():
                self.duration_corrector.fit(X_stack[m],
                                            np.asarray(duration_y)[m] - nn_dur[m])
                fitted.append("duration")

        self._fitted_correctors = bool(fitted)
        return fitted

    # ---------- inference ----------
    def predict(self, soil_x, climate_x):
        """
        Returns (crop_idx, crop_probs, yield_pred, duration_pred).

        yield_pred / duration_pred are None when the model was never trained
        on that target. Returning None rather than the untrained head's output
        is deliberate: it forces the caller (Stage 5) to fall back to the
        reference economics table and to label the estimate honestly, instead
        of presenting random-initialised noise as a model prediction.
        """
        logits, nn_yield, nn_dur, embeddings = self._forward_numpy(soil_x, climate_x)

        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        crop_probs = exp / exp.sum(axis=1, keepdims=True)
        crop_idx = np.argmax(crop_probs, axis=1)

        X_stack = None
        if self._fitted_correctors:
            X_stack = np.concatenate([embeddings, soil_x, climate_x], axis=1)

        yield_pred = None
        if self.trained_yield:
            yield_pred = nn_yield
            if X_stack is not None:
                yield_pred = yield_pred + self.yield_corrector.predict(X_stack)
            yield_pred = np.clip(yield_pred, 0.1, None)

        dur_pred = None
        if self.trained_duration:
            dur_pred = nn_dur
            if X_stack is not None:
                dur_pred = dur_pred + self.duration_corrector.predict(X_stack)
            dur_pred = np.clip(dur_pred, 20, None)

        return crop_idx, crop_probs, yield_pred, dur_pred
