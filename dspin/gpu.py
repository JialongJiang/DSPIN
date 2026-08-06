#!/usr/bin/env python

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


class CupyBackendError(RuntimeError):
    """Raised when the optional CuPy network backend cannot run."""


def require_cupy(device: int = 0):
    try:
        import cupy as cp
    except ImportError as exc:
        raise CupyBackendError(
            "backend='cupy' requires CuPy in the active Python environment. "
            "Install a CUDA-matched CuPy build, for example `pip install dspin[gpu]` "
            "or `pip install cupy-cuda12x`."
        ) from exc

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise CupyBackendError("CuPy is installed, but no CUDA runtime/device is available.") from exc

    if device_count < 1:
        raise CupyBackendError("CuPy found zero CUDA devices.")
    if device < 0 or device >= device_count:
        raise CupyBackendError(f"CUPY_DEVICE={device} is invalid; visible CUDA devices: {device_count}.")

    cp.cuda.Device(device).use()
    return cp


def normalize_dtype(dtype: str) -> np.dtype:
    normalized = str(dtype).lower()
    if normalized not in {"float64", "float32"}:
        raise CupyBackendError("CUPY_DTYPE must be either float64 or float32.")
    return np.dtype(normalized)


def normalize_optimizer_semantics(value: str) -> str:
    normalized = str(value).lower()
    if normalized not in {"python", "matlab"}:
        raise CupyBackendError("cupy_optimizer_semantics must be either 'python' or 'matlab'.")
    return normalized


def _param(train_dat: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in train_dat:
            return train_dat[name]
    return default


def _validate_raw_data(raw_data: Sequence[np.ndarray]) -> Tuple[int, int]:
    if raw_data is None or len(raw_data) == 0:
        raise CupyBackendError("CuPy network inference received empty raw_data.")
    num_spin = None
    total_cells = 0
    for idx, state in enumerate(raw_data):
        arr = np.asarray(state)
        if arr.ndim != 2:
            raise CupyBackendError(f"raw_data sample {idx} must be 2D, got shape {arr.shape}.")
        if arr.shape[1] == 0:
            raise CupyBackendError(f"raw_data sample {idx} has zero cells.")
        if num_spin is None:
            num_spin = arr.shape[0]
        elif arr.shape[0] != num_spin:
            raise CupyBackendError(
                f"raw_data sample {idx} has {arr.shape[0]} spins, expected {num_spin}."
            )
        total_cells += arr.shape[1]
    return int(num_spin), int(total_cells)


def _auto_chunk_cells(cp, num_spin: int, dtype: np.dtype) -> int:
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_value = dtype.itemsize
    bytes_per_cell = max(1, 6 * num_spin * bytes_per_value)
    chunk = int((0.20 * free_bytes) // bytes_per_cell)
    return max(128, min(chunk, 100_000))


def _resolve_chunk_cells(cp, chunk_cells: Any, num_spin: int, dtype: np.dtype) -> int:
    if chunk_cells is None or str(chunk_cells).lower() == "auto":
        return _auto_chunk_cells(cp, num_spin, dtype)
    try:
        chunk = int(chunk_cells)
    except (TypeError, ValueError) as exc:
        raise CupyBackendError("CUPY_CHUNK_CELLS must be a positive integer or 'auto'.") from exc
    if chunk < 1:
        raise CupyBackendError("CUPY_CHUNK_CELLS must be at least 1.")
    return chunk


def _l1_base_schedule(num_epoch: int) -> np.ndarray:
    l1_base_start = 0.02
    l1_base_end = 1e-3
    middle = np.logspace(np.log10(l1_base_start), np.log10(l1_base_end), max(2, num_epoch // 2))
    schedule = np.concatenate(
        [
            np.full(num_epoch // 4 + 1, l1_base_start),
            middle,
            np.full(num_epoch // 4 + 1, l1_base_end),
        ]
    )
    return schedule[:num_epoch]


def _adam_update(cp, gradient, m, v, counter: int, stepsz: float):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** counter)
    v_hat = v / (1 - beta2 ** counter)
    update = stepsz * m_hat / (cp.sqrt(v_hat) + epsilon)
    return update, m, v


def _compute_relative_response_cp(cp, cur_h, if_control: np.ndarray, batch_index: np.ndarray):
    if_control = np.asarray(if_control).astype(bool).reshape(-1)
    batch_index = np.asarray(batch_index).reshape(-1)
    if if_control.size != cur_h.shape[1] or batch_index.size != cur_h.shape[1]:
        raise CupyBackendError("if_control and batch_index must match the number of DSPIN samples.")

    relative_h = cp.zeros_like(cur_h)
    if np.all(~if_control):
        all_controls_mean = cp.mean(cur_h, axis=1)
    else:
        all_controls_mean = cp.mean(cur_h[:, if_control], axis=1)

    for current_batch in np.unique(batch_index):
        batch_samples_idx = np.where(batch_index == current_batch)[0]
        batch_controls_idx = batch_samples_idx[if_control[batch_samples_idx]]
        if batch_controls_idx.size > 0:
            batch_controls_mean = cp.mean(cur_h[:, batch_controls_idx], axis=1)
        else:
            batch_controls_mean = all_controls_mean
        relative_h[:, batch_samples_idx] = cur_h[:, batch_samples_idx] - batch_controls_mean[:, None]
    return relative_h


def _average_batch_control_cp(cp, hgrad, if_control: np.ndarray, batch_index: np.ndarray):
    hgrad_raw = hgrad
    hgrad = hgrad_raw.copy()
    for current_batch in np.unique(batch_index):
        batch_samples_idx = np.where(batch_index == current_batch)[0]
        batch_controls_idx = batch_samples_idx[if_control[batch_samples_idx]]
        if batch_controls_idx.size > 0:
            batch_mean = cp.mean(hgrad_raw[:, batch_samples_idx], axis=1)
            hgrad[:, batch_controls_idx] = batch_mean[:, None]
    return hgrad


def _pseudo_gradient_sample_cupy(
    cp,
    cur_j,
    j_no_diag,
    h_vec,
    cur_state_cpu: np.ndarray,
    directed: bool,
    dtype: np.dtype,
    chunk_cells: int,
):
    cur_state_cpu = np.asarray(cur_state_cpu)
    num_spin, num_cells = cur_state_cpu.shape
    if num_cells == 0:
        raise CupyBackendError("Cannot compute a DSPIN gradient for a sample with zero cells.")

    cp_dtype = cp.float64 if dtype == np.dtype("float64") else cp.float32
    diag_j = cp.diag(cur_j)[:, None]
    j_grad_sum = cp.zeros((num_spin, num_spin), dtype=cp_dtype)
    diag_grad_sum = cp.zeros(num_spin, dtype=cp_dtype)
    h_grad_sum = cp.zeros(num_spin, dtype=cp_dtype)

    for start in range(0, num_cells, chunk_cells):
        end = min(start + chunk_cells, num_cells)
        state = cp.asarray(cur_state_cpu[:, start:end], dtype=cp_dtype)
        eff_h = j_no_diag @ state + h_vec
        term1 = cp.exp(diag_j + eff_h)
        term2 = cp.exp(diag_j - eff_h)
        denom = term1 + term2 + 1.0
        h_eff_grad = state - (term1 - term2) / denom
        diag_term = state * state - (term1 + term2) / denom

        j_grad_sum += h_eff_grad @ state.T
        diag_grad_sum += cp.sum(diag_term, axis=1)
        h_grad_sum += cp.sum(h_eff_grad, axis=1)

    j_grad = j_grad_sum / num_cells
    diag_idx = cp.arange(num_spin)
    j_grad[diag_idx, diag_idx] = diag_grad_sum / num_cells
    h_grad = h_grad_sum / num_cells
    if not directed:
        j_grad = 0.5 * (j_grad + j_grad.T)
    return -j_grad, -h_grad


def _snapshot(cp, counter: int, grad_norm: float, cur_j, cur_h, mjj, vjj, mhh, vhh) -> Dict[str, Any]:
    return {
        "counter": counter,
        "grad_norm": grad_norm,
        "cur_j": cp.asnumpy(cur_j),
        "cur_h": cp.asnumpy(cur_h),
        "mjj": cp.asnumpy(mjj),
        "vjj": cp.asnumpy(vjj),
        "mhh": cp.asnumpy(mhh),
        "vhh": cp.asnumpy(vhh),
    }


def _find_snapshot(snapshots, counter: int) -> Dict[str, Any]:
    item = snapshots.get(counter)
    if item is not None:
        return item
    raise CupyBackendError(f"Internal error: missing optimizer snapshot for epoch {counter}.")


def _find_program_regulator_snapshot(snapshots, counter: int) -> Dict[str, Any]:
    item = snapshots.get(counter)
    if item is not None:
        return item
    raise CupyBackendError(f"Internal error: missing program-regulator snapshot for epoch {counter}.")


def _record_bounded_snapshot(snapshots, snapshot: Dict[str, Any], history_len: int) -> None:
    """Keep one checkpoint per epoch so replayed epochs cannot evict backtracking targets."""
    snapshots[snapshot["counter"]] = snapshot
    while len(snapshots) > history_len:
        del snapshots[min(snapshots)]


def _save_log_npz(cp, save_path: str, cur_j, cur_h, rec_jgrad_sum_norm: np.ndarray) -> None:
    if not save_path:
        return
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        path / "network_cupy_log.npz",
        cur_j=cp.asnumpy(cur_j),
        cur_h=cp.asnumpy(cur_h),
        rec_jgrad_sum_norm=rec_jgrad_sum_norm,
    )


def _save_program_regulator_log_npz(
    cp,
    save_path: str,
    cur_interaction,
    cur_selfj,
    cur_selfh,
    rec_grad: np.ndarray,
) -> None:
    if not save_path:
        return
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(
        path / "program_regulators_cupy_log.npz",
        cur_interaction=cp.asnumpy(cur_interaction),
        cur_selfj=cp.asnumpy(cur_selfj),
        cur_selfh=cp.asnumpy(cur_selfh),
        rec_grad=rec_grad,
    )


def _validate_program_regulator_data(
    gene_states: Sequence[np.ndarray],
    program_states: Sequence[np.ndarray],
) -> Tuple[int, int, int]:
    if gene_states is None or program_states is None or len(gene_states) == 0:
        raise CupyBackendError("CuPy program-regulator discovery received empty state data.")
    if len(gene_states) != len(program_states):
        raise CupyBackendError(
            f"gene_states has {len(gene_states)} samples but program_states has {len(program_states)}."
        )

    num_gene = None
    num_program = None
    total_cells = 0
    for idx, (gene_state, program_state) in enumerate(zip(gene_states, program_states)):
        gene_arr = np.asarray(gene_state)
        program_arr = np.asarray(program_state)
        if gene_arr.ndim != 2 or program_arr.ndim != 2:
            raise CupyBackendError(
                f"Sample {idx} states must be 2D; got {gene_arr.shape} and {program_arr.shape}."
            )
        if gene_arr.shape[1] == 0 or program_arr.shape[1] == 0:
            raise CupyBackendError(f"Sample {idx} has zero cells.")
        if gene_arr.shape[1] != program_arr.shape[1]:
            raise CupyBackendError(
                f"Sample {idx} gene/program cell counts differ: {gene_arr.shape[1]} vs {program_arr.shape[1]}."
            )
        if num_gene is None:
            num_gene = gene_arr.shape[0]
            num_program = program_arr.shape[0]
        elif gene_arr.shape[0] != num_gene or program_arr.shape[0] != num_program:
            raise CupyBackendError(
                f"Sample {idx} has shape {gene_arr.shape}/{program_arr.shape}; "
                f"expected ({num_gene}, cells)/({num_program}, cells)."
            )
        total_cells += gene_arr.shape[1]
    return int(num_gene), int(num_program), int(total_cells)


def _auto_program_regulator_chunk_cells(cp, num_gene: int, num_program: int, dtype: np.dtype) -> int:
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_value = dtype.itemsize
    bytes_per_cell = max(1, (2 * num_gene + 6 * num_program) * bytes_per_value)
    chunk = int((0.20 * free_bytes) // bytes_per_cell)
    return max(128, min(chunk, 250_000))


def _resolve_program_regulator_chunk_cells(
    cp,
    chunk_cells: Any,
    num_gene: int,
    num_program: int,
    dtype: np.dtype,
) -> int:
    if chunk_cells is None or str(chunk_cells).lower() == "auto":
        return _auto_program_regulator_chunk_cells(cp, num_gene, num_program, dtype)
    try:
        chunk = int(chunk_cells)
    except (TypeError, ValueError) as exc:
        raise CupyBackendError("PROGRAM_REGULATOR_CUPY_CHUNK_CELLS must be a positive integer or 'auto'.") from exc
    if chunk < 1:
        raise CupyBackendError("PROGRAM_REGULATOR_CUPY_CHUNK_CELLS must be at least 1.")
    return chunk


def _program_regulator_sample_gradient_cupy(
    cp,
    cur_interaction,
    cur_selfj,
    cur_selfh,
    gene_state_cpu: np.ndarray,
    program_state_cpu: np.ndarray,
    dtype: np.dtype,
    chunk_cells: int,
):
    gene_state_cpu = np.asarray(gene_state_cpu)
    program_state_cpu = np.asarray(program_state_cpu)
    num_gene, num_cells = gene_state_cpu.shape
    num_program = program_state_cpu.shape[0]
    cp_dtype = cp.float64 if dtype == np.dtype("float64") else cp.float32

    interaction_grad_sum = cp.zeros((num_gene, num_program), dtype=cp_dtype)
    selfj_grad_sum = cp.zeros(num_program, dtype=cp_dtype)
    selfh_grad_sum = cp.zeros(num_program, dtype=cp_dtype)

    for start in range(0, num_cells, chunk_cells):
        end = min(start + chunk_cells, num_cells)
        gene_state = cp.asarray(gene_state_cpu[:, start:end], dtype=cp_dtype)
        program_state = cp.asarray(program_state_cpu[:, start:end], dtype=cp_dtype)
        effective_h = gene_state.T @ cur_interaction + cur_selfh[None, :]
        term1 = cp.exp(cur_selfj[:, None] + effective_h.T)
        term2 = cp.exp(cur_selfj[:, None] - effective_h.T)
        denom = term1 + term2 + 1.0
        selfj_grad = program_state * program_state - (term1 + term2) / denom
        h_eff_grad = program_state - (term1 - term2) / denom

        interaction_grad_sum += gene_state @ h_eff_grad.T
        selfj_grad_sum += cp.sum(selfj_grad, axis=1)
        selfh_grad_sum += cp.sum(h_eff_grad, axis=1)

    return (
        interaction_grad_sum / num_cells,
        selfj_grad_sum / num_cells,
        selfh_grad_sum / num_cells,
    )


def _program_regulator_snapshot(
    cp,
    counter: int,
    grad_norm: float,
    cur_interaction,
    cur_selfj,
    cur_selfh,
    m_interaction,
    v_interaction,
    m_selfj,
    v_selfj,
    m_selfh,
    v_selfh,
) -> Dict[str, Any]:
    return {
        "counter": counter,
        "grad_norm": grad_norm,
        "cur_interaction": cp.asnumpy(cur_interaction),
        "cur_selfj": cp.asnumpy(cur_selfj),
        "cur_selfh": cp.asnumpy(cur_selfh),
        "m_interaction": cp.asnumpy(m_interaction),
        "v_interaction": cp.asnumpy(v_interaction),
        "m_selfj": cp.asnumpy(m_selfj),
        "v_selfj": cp.asnumpy(v_selfj),
        "m_selfh": cp.asnumpy(m_selfh),
        "v_selfh": cp.asnumpy(v_selfh),
    }


def learn_program_regulators_cupy(
    gene_states: Sequence[np.ndarray],
    program_states: Sequence[np.ndarray],
    train_dat: Dict[str, Any],
    device: int = 0,
    dtype: str = "float64",
    chunk_cells: Any = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    cp = require_cupy(device)
    dtype_np = normalize_dtype(dtype)
    cp_dtype = cp.float64 if dtype_np == np.dtype("float64") else cp.float32
    num_gene, num_program, _ = _validate_program_regulator_data(gene_states, program_states)
    chunk_cells = _resolve_program_regulator_chunk_cells(
        cp,
        chunk_cells,
        num_gene,
        num_program,
        dtype_np,
    )

    num_round = len(gene_states)
    num_epoch = int(_param(train_dat, "num_epoch", "epoch", default=500))
    if num_epoch < 1:
        raise CupyBackendError("PROGRAM_REGULATOR_NUM_EPOCH must be at least 1.")
    stepsz = float(_param(train_dat, "stepsz", default=0.01))
    rec_gap = int(_param(train_dat, "rec_gap", default=10))
    backtrack_gap = int(_param(train_dat, "backtrack_gap", default=40))
    backtrack_tol = int(_param(train_dat, "backtrack_tol", default=5))
    save_log = bool(_param(train_dat, "save_log", default=False))
    save_path = str(_param(train_dat, "save_path", default=""))

    cur_interaction = cp.asarray(
        train_dat.get("cur_interaction", np.zeros((num_gene, num_program))),
        dtype=cp_dtype,
    )
    cur_selfj = cp.asarray(train_dat.get("cur_selfj", np.zeros(num_program)), dtype=cp_dtype)
    cur_selfh = cp.asarray(train_dat.get("cur_selfh", np.zeros(num_program)), dtype=cp_dtype)
    if cur_interaction.shape != (num_gene, num_program):
        raise CupyBackendError(
            f"cur_interaction has shape {cur_interaction.shape}; expected {(num_gene, num_program)}."
        )
    if cur_selfj.shape != (num_program,) or cur_selfh.shape != (num_program,):
        raise CupyBackendError("cur_selfj and cur_selfh must be 1D arrays with one value per program.")

    state_sizes = np.array([np.asarray(state).shape[1] for state in gene_states], dtype=float)
    sample_weight = np.sqrt(state_sizes)
    sample_weight = sample_weight / sample_weight.sum()
    sample_weight_gpu = cp.asarray(sample_weight, dtype=cp_dtype)

    m_interaction = cp.zeros_like(cur_interaction)
    v_interaction = cp.zeros_like(cur_interaction)
    m_selfj = cp.zeros_like(cur_selfj)
    v_selfj = cp.zeros_like(cur_selfj)
    m_selfh = cp.zeros_like(cur_selfh)
    v_selfh = cp.zeros_like(cur_selfh)
    rec_grad = np.full(num_epoch, np.inf, dtype=float)
    l1_base_array = _l1_base_schedule(num_epoch)
    list_step = list(np.arange(num_epoch, 0, -rec_gap)[::-1])
    plot_counter = 0
    cur_backtrack = 0
    history_len = max(50, backtrack_gap + 1)
    snapshots = {}

    lambda_l1_interaction = float(
        _param(train_dat, "lambda_l1_interaction", "lambda_l1_j", default=0)
    )
    lambda_l2_interaction = float(
        _param(train_dat, "lambda_l2_interaction", "lambda_l2_j", default=0)
    )

    print(
        f"Using CuPy program-regulator discovery on device {device}; "
        f"dtype={dtype_np.name}; chunk_cells={chunk_cells}",
        flush=True,
    )

    counter = 1
    while counter <= num_epoch:
        l1_base = float(l1_base_array[counter - 1])
        interaction_grad = cp.zeros_like(cur_interaction)
        selfj_grad = cp.zeros_like(cur_selfj)
        selfh_grad = cp.zeros_like(cur_selfh)

        for kk, (gene_state, program_state) in enumerate(zip(gene_states, program_states)):
            cur_interaction_grad, cur_selfj_grad, cur_selfh_grad = _program_regulator_sample_gradient_cupy(
                cp,
                cur_interaction,
                cur_selfj,
                cur_selfh,
                gene_state,
                program_state,
                dtype_np,
                chunk_cells,
            )
            interaction_grad -= cur_interaction_grad * sample_weight_gpu[kk]
            selfj_grad -= cur_selfj_grad * sample_weight_gpu[kk]
            selfh_grad -= cur_selfh_grad * sample_weight_gpu[kk]

        if lambda_l1_interaction > 0:
            interaction_grad += lambda_l1_interaction * cp.clip(cur_interaction / l1_base, -1, 1)
        if lambda_l2_interaction > 0:
            interaction_grad += lambda_l2_interaction * cur_interaction

        update_val, m_interaction, v_interaction = _adam_update(
            cp,
            interaction_grad,
            m_interaction,
            v_interaction,
            counter,
            stepsz,
        )
        cur_interaction = cur_interaction - update_val
        update_val, m_selfj, v_selfj = _adam_update(cp, selfj_grad, m_selfj, v_selfj, counter, stepsz)
        cur_selfj = cur_selfj - update_val
        update_val, m_selfh, v_selfh = _adam_update(cp, selfh_grad, m_selfh, v_selfh, counter, stepsz)
        cur_selfh = cur_selfh - update_val

        grad_norm = float(cp.linalg.norm(interaction_grad).get())
        rec_grad[counter - 1] = grad_norm
        _record_bounded_snapshot(
            snapshots,
            _program_regulator_snapshot(
                cp,
                counter,
                grad_norm,
                cur_interaction,
                cur_selfj,
                cur_selfh,
                m_interaction,
                v_interaction,
                m_selfj,
                v_selfj,
                m_selfh,
                v_selfh,
            ),
            history_len,
        )

        if plot_counter < len(list_step) and counter == list_step[plot_counter]:
            pbar_pct = round(100 * (plot_counter + 1) / len(list_step))
            print(f"{counter}/{num_epoch} {pbar_pct}% Network Gradient={grad_norm:.6f}", flush=True)
            if save_log:
                _save_program_regulator_log_npz(
                    cp,
                    save_path,
                    cur_interaction,
                    cur_selfj,
                    cur_selfh,
                    rec_grad,
                )
            plot_counter += 1

        if_backtrack = False
        if counter > backtrack_gap:
            prev_grad = rec_grad[counter - 1 - backtrack_gap]
            if np.isfinite(prev_grad) and rec_grad[counter - 1] > 1.5 * prev_grad:
                if_backtrack = True
        if counter > 2 * backtrack_gap and counter % backtrack_gap == 0:
            est_cur_grad = rec_grad[counter - 1 - backtrack_gap : counter - 1].mean()
            est_prev_grad = rec_grad[counter - 1 - 2 * backtrack_gap : counter - 1 - backtrack_gap].mean()
            if np.isfinite(est_cur_grad) and np.isfinite(est_prev_grad) and est_cur_grad > 1.05 * est_prev_grad:
                if_backtrack = True

        if if_backtrack:
            print(f"Backtracking at epoch {counter}", flush=True)
            cur_backtrack += 1
            target_counter = counter - backtrack_gap
            target = _find_program_regulator_snapshot(snapshots, target_counter)
            cur_interaction = cp.asarray(target["cur_interaction"], dtype=cp_dtype)
            cur_selfj = cp.asarray(target["cur_selfj"], dtype=cp_dtype)
            cur_selfh = cp.asarray(target["cur_selfh"], dtype=cp_dtype)
            m_interaction = cp.asarray(target["m_interaction"], dtype=cp_dtype)
            v_interaction = cp.asarray(target["v_interaction"], dtype=cp_dtype)
            m_selfj = cp.asarray(target["m_selfj"], dtype=cp_dtype)
            v_selfj = cp.asarray(target["v_selfj"], dtype=cp_dtype)
            m_selfh = cp.asarray(target["m_selfh"], dtype=cp_dtype)
            v_selfh = cp.asarray(target["v_selfh"], dtype=cp_dtype)
            snapshots = {
                epoch: item for epoch, item in snapshots.items() if epoch <= target_counter
            }
            rec_grad[target_counter:] = np.inf
            counter = target_counter
            stepsz = stepsz / 4
            if cur_backtrack > backtrack_tol:
                print(f"Backtracking more than {backtrack_tol} times, stop training.", flush=True)
                break
            continue

        counter += 1

    finite_pos = np.where(np.isfinite(rec_grad))[0]
    if finite_pos.size == 0:
        raise CupyBackendError("CuPy program-regulator optimizer did not produce finite gradients.")
    tail_pos = finite_pos[max(0, finite_pos.size - 50) :]
    best_counter = int(tail_pos[np.argmin(rec_grad[tail_pos])]) + 1
    best = _find_program_regulator_snapshot(snapshots, best_counter)
    train_log = {
        "program_regulator_gradient": rec_grad,
        "backend": "cupy",
        "device": device,
        "dtype": dtype_np.name,
        "chunk_cells": chunk_cells,
        "best_epoch": best_counter,
    }
    return (
        best["cur_interaction"],
        best["cur_selfj"],
        best["cur_selfh"],
        train_log,
    )


def learn_network_cupy(
    raw_data: Sequence[np.ndarray],
    train_dat: Dict[str, Any],
    device: int = 0,
    dtype: str = "float64",
    chunk_cells: Any = "auto",
    optimizer_semantics: str = "python",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    cp = require_cupy(device)
    dtype_np = normalize_dtype(dtype)
    optimizer_semantics = normalize_optimizer_semantics(optimizer_semantics)
    cp_dtype = cp.float64 if dtype_np == np.dtype("float64") else cp.float32
    num_spin, _ = _validate_raw_data(raw_data)
    chunk_cells = _resolve_chunk_cells(cp, chunk_cells, num_spin, dtype_np)

    method = _param(train_dat, "method", default="pseudo_likelihood")
    if method != "pseudo_likelihood":
        raise CupyBackendError("The CuPy backend currently supports only pseudo_likelihood inference.")

    num_round = len(raw_data)
    num_epoch = int(_param(train_dat, "num_epoch", "epoch", default=400))
    if num_epoch < 1:
        raise CupyBackendError("num_epoch must be at least 1.")
    stepsz = float(_param(train_dat, "stepsz", default=0.05))
    rec_gap = int(_param(train_dat, "rec_gap", default=10))
    backtrack_gap = int(_param(train_dat, "backtrack_gap", default=40))
    backtrack_tol = int(_param(train_dat, "backtrack_tol", default=5))
    directed = bool(_param(train_dat, "directed", default=True))
    save_log = bool(_param(train_dat, "save_log", default=False))
    save_path = str(_param(train_dat, "save_path", default=""))

    cur_j = cp.asarray(train_dat.get("cur_j", np.zeros((num_spin, num_spin))), dtype=cp_dtype)
    cur_h = cp.asarray(train_dat.get("cur_h", np.zeros((num_spin, num_round))), dtype=cp_dtype)
    if cur_j.shape != (num_spin, num_spin):
        raise CupyBackendError(f"cur_j has shape {cur_j.shape}; expected {(num_spin, num_spin)}.")
    if cur_h.shape != (num_spin, num_round):
        raise CupyBackendError(f"cur_h has shape {cur_h.shape}; expected {(num_spin, num_round)}.")

    state_sizes = np.array([np.asarray(state).shape[1] for state in raw_data], dtype=float)
    sample_weight = np.sqrt(state_sizes)
    sample_weight = sample_weight / sample_weight.sum()
    sample_weight_gpu = cp.asarray(sample_weight, dtype=cp_dtype)

    mjj = cp.zeros_like(cur_j)
    vjj = cp.zeros_like(cur_j)
    mhh = cp.zeros_like(cur_h)
    vhh = cp.zeros_like(cur_h)
    rec_jgrad_sum_norm = np.full(num_epoch, np.inf, dtype=float)
    l1_base_array = _l1_base_schedule(num_epoch)
    list_step = list(np.arange(num_epoch, 0, -rec_gap)[::-1])
    plot_counter = 0
    cur_backtrack = 0
    history_len = max(50, backtrack_gap + 1)
    snapshots = {}

    lambda_l1_j = float(_param(train_dat, "lambda_l1_j", "lam_l1j", default=0))
    lambda_l2_j = float(_param(train_dat, "lambda_l2_j", "lam_l2j", default=0))
    lambda_l1_h = float(_param(train_dat, "lambda_l1_h", "lam_l1h", default=0))
    lambda_l2_h = float(_param(train_dat, "lambda_l2_h", "lam_l2h", default=0))
    lambda_l1_j_prior_mask = float(_param(train_dat, "lambda_l1_j_prior_mask", "lam_l1j_prior", default=0))
    lambda_l2_j_prior_mask = float(_param(train_dat, "lambda_l2_j_prior_mask", default=0))
    lambda_l2_j_prior = float(_param(train_dat, "lambda_l2_j_prior", default=0))
    j_prior_mask = _param(train_dat, "j_prior_mask", "prior_network", default=None)
    j_prior_mask_gpu = cp.asarray(j_prior_mask, dtype=cp_dtype) if j_prior_mask is not None else None
    j_prior = _param(train_dat, "j_prior", default=None)
    j_prior_gpu = cp.asarray(j_prior, dtype=cp_dtype) if j_prior is not None else None

    perturb_matrix = _param(train_dat, "perturb_matrix", default=None)
    lambda_l2_h_perturb = float(_param(train_dat, "lambda_l2_h_rela_prior", "lam_l2h_perturb", default=0))
    lambda_l1_h_rela = float(_param(train_dat, "lambda_l1_h_rela", default=0))
    lambda_l2_h_rela = float(_param(train_dat, "lambda_l2_h_rela", default=0))
    if perturb_matrix is not None:
        perturb_matrix_gpu = cp.asarray(perturb_matrix, dtype=cp_dtype)
        if_control = np.asarray(train_dat["if_control"]).astype(bool)
        batch_index = np.asarray(train_dat["batch_index"])
    else:
        perturb_matrix_gpu = None
        if_control = None
        batch_index = None

    print(
        f"Using CuPy pseudo_likelihood network inference on device {device}; "
        f"dtype={dtype_np.name}; chunk_cells={chunk_cells}; semantics={optimizer_semantics}",
        flush=True,
    )

    counter = 1
    diag_idx = cp.arange(num_spin)
    while counter <= num_epoch:
        print(f"{counter} ", end="", flush=True)
        l1_base = 1e-3 if optimizer_semantics == "python" else float(l1_base_array[counter - 1])
        rec_jgrad_sum = cp.zeros_like(cur_j)
        rec_hgrad = cp.zeros_like(cur_h)
        j_no_diag = cur_j.copy()
        j_no_diag[diag_idx, diag_idx] = 0.0

        for kk, state_cpu in enumerate(raw_data):
            corr_grad, mean_grad = _pseudo_gradient_sample_cupy(
                cp,
                cur_j,
                j_no_diag,
                cur_h[:, kk : kk + 1],
                state_cpu,
                directed,
                dtype_np,
                chunk_cells,
            )
            if optimizer_semantics == "python":
                rec_jgrad_sum += corr_grad
            else:
                rec_jgrad_sum += corr_grad * sample_weight_gpu[kk]
            rec_hgrad[:, kk] = mean_grad

        j_regularization_scale = num_round if optimizer_semantics == "python" else 1
        if lambda_l1_j > 0:
            rec_jgrad_sum += j_regularization_scale * lambda_l1_j * cp.clip(cur_j / l1_base, -1, 1)
        if lambda_l2_j > 0:
            rec_jgrad_sum += j_regularization_scale * lambda_l2_j * cur_j
        if lambda_l1_h > 0:
            rec_hgrad += lambda_l1_h * cp.clip(cur_h / l1_base, -1, 1)
        if lambda_l2_h > 0:
            rec_hgrad += lambda_l2_h * cur_h
        if j_prior_gpu is not None and lambda_l2_j_prior > 0:
            rec_jgrad_sum += j_regularization_scale * lambda_l2_j_prior * (cur_j - j_prior_gpu)
        if j_prior_mask_gpu is not None and lambda_l2_j_prior_mask > 0:
            prior_mask = j_prior_mask_gpu == 0
            rec_jgrad_sum += j_regularization_scale * lambda_l2_j_prior_mask * cur_j * prior_mask
        if j_prior_mask_gpu is not None and lambda_l1_j_prior_mask > 0:
            prior_mask = j_prior_mask_gpu == 0
            rec_jgrad_sum += (
                j_regularization_scale
                * lambda_l1_j_prior_mask
                * cp.clip((cur_j * prior_mask) / l1_base, -1, 1)
            )
        if perturb_matrix_gpu is not None and lambda_l2_h_perturb > 0:
            if optimizer_semantics == "python":
                rec_hgrad = _average_batch_control_cp(cp, rec_hgrad, if_control, batch_index)
            relative_h = _compute_relative_response_cp(cp, cur_h, if_control, batch_index)
            rec_hgrad += lambda_l2_h_perturb * (relative_h - perturb_matrix_gpu)
        elif if_control is not None:
            relative_h = _compute_relative_response_cp(cp, cur_h, if_control, batch_index)
        else:
            relative_h = None
        if relative_h is not None and lambda_l1_h_rela > 0:
            rec_hgrad += lambda_l1_h_rela * cp.clip(relative_h / 1e-3, -1, 1)
        if relative_h is not None and lambda_l2_h_rela > 0:
            rec_hgrad += lambda_l2_h_rela * relative_h

        update_j, mjj, vjj = _adam_update(cp, rec_jgrad_sum, mjj, vjj, counter, stepsz)
        update_h, mhh, vhh = _adam_update(cp, rec_hgrad, mhh, vhh, counter, stepsz)
        cur_j = cur_j - update_j
        cur_h = cur_h - update_h

        grad_norm = float(cp.linalg.norm(rec_jgrad_sum).get())
        rec_jgrad_sum_norm[counter - 1] = grad_norm
        _record_bounded_snapshot(
            snapshots,
            _snapshot(cp, counter, grad_norm, cur_j, cur_h, mjj, vjj, mhh, vhh),
            history_len,
        )

        if plot_counter < len(list_step) and counter == list_step[plot_counter]:
            percent = round(100 * (plot_counter + 1) / len(list_step))
            print(f"{percent}% gradient={grad_norm:g}", flush=True)
            if save_log:
                _save_log_npz(cp, save_path, cur_j, cur_h, rec_jgrad_sum_norm)
            plot_counter += 1

        if_backtrack = False
        if counter > backtrack_gap:
            previous = rec_jgrad_sum_norm[counter - backtrack_gap - 1]
            threshold = 1.5 if optimizer_semantics == "python" else 2.0
            if np.isfinite(previous) and grad_norm > threshold * previous:
                if_backtrack = True
        if (
            optimizer_semantics == "python"
            and counter > 2 * backtrack_gap
            and counter % backtrack_gap == 0
        ):
            est_cur_grad = rec_jgrad_sum_norm[counter - 1 - backtrack_gap : counter - 1].mean()
            est_prev_grad = rec_jgrad_sum_norm[
                counter - 1 - 2 * backtrack_gap : counter - 1 - backtrack_gap
            ].mean()
            if np.isfinite(est_cur_grad) and np.isfinite(est_prev_grad) and est_cur_grad > 1.05 * est_prev_grad:
                if_backtrack = True
        if if_backtrack:
            target_counter = counter - backtrack_gap
            target = _find_snapshot(snapshots, target_counter)
            cur_j = cp.asarray(target["cur_j"], dtype=cp_dtype)
            cur_h = cp.asarray(target["cur_h"], dtype=cp_dtype)
            mjj = cp.asarray(target["mjj"], dtype=cp_dtype)
            vjj = cp.asarray(target["vjj"], dtype=cp_dtype)
            mhh = cp.asarray(target["mhh"], dtype=cp_dtype)
            vhh = cp.asarray(target["vhh"], dtype=cp_dtype)
            snapshots = {
                epoch: item for epoch, item in snapshots.items() if epoch <= target_counter
            }
            if optimizer_semantics == "python":
                counter = target_counter + 1
                stepsz = stepsz / 4
            else:
                rec_jgrad_sum_norm[target_counter:] = np.inf
                counter = target_counter
                stepsz = stepsz / 2
            cur_backtrack += 1
            print(f"backtrack {stepsz}", flush=True)
            if cur_backtrack > backtrack_tol:
                print("backtrack tolerance reached", flush=True)
                break
            continue

        counter += 1

    print("", flush=True)
    finite_pos = np.where(np.isfinite(rec_jgrad_sum_norm))[0]
    if finite_pos.size == 0:
        raise CupyBackendError("CuPy optimizer did not produce any finite gradient values.")
    tail_pos = finite_pos[max(0, finite_pos.size - 50) :]
    best_counter = int(tail_pos[np.argmin(rec_jgrad_sum_norm[tail_pos])]) + 1
    best = _find_snapshot(snapshots, best_counter)
    train_log = {
        "network_gradient": rec_jgrad_sum_norm,
        "backend": "cupy",
        "device": device,
        "dtype": dtype_np.name,
        "chunk_cells": chunk_cells,
        "optimizer_semantics": optimizer_semantics,
        "best_epoch": best_counter,
    }
    return best["cur_j"], best["cur_h"], train_log
