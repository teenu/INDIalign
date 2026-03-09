#!/usr/bin/env python3
"""
Multithreaded local TM-score validation for Stanford RNA 3D Folding (Competition 2).

This is a parallelized, from-scratch implementation of the exact scoring logic
in `tm-score-permutechains.ipynb`, with per-target parallelism.

Key points:
- Uses USalign with `-atom " C1'"`
- For chain mapping: `-TMscore 0 -mm 1 -ter 0`
- Final scoring uses the *second* TM-score (normalized by native length)
- Multicopy handling and chain permutation strictly mirror the notebook

Usage:
    python local_validation_mt.py submission.csv
    python local_validation_mt.py submission.csv --workers 16
    python local_validation_mt.py submission.csv --validation /path/to/validation_labels.csv --usalign /path/to/USalign
"""

import os
import re
import argparse
import tempfile
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------
# USalign helpers (as in tm-score-permutechains.ipynb)
# ------------------------------------------------------------

def parse_tmscore_output(output: str) -> float:
    matches = re.findall(r'TM-score=\s+([\d.]+)', output)
    if len(matches) < 2:
        raise ValueError('No TM score found in USalign output')
    return float(matches[1])


def run_usalign_raw(predicted_pdb: str, native_pdb: str, usalign_bin='USalign', align_sequence=False, tmscore=None) -> str:
    cmd = [usalign_bin, predicted_pdb, native_pdb, '-atom', " C1'"]
    if tmscore is not None:
        cmd += ['-TMscore', str(tmscore)]
        if int(tmscore) == 0:
            cmd += ['-mm', '1', '-ter', '0']
    elif not align_sequence:
        cmd += ['-TMscore', '1']

    # Use subprocess.run for safety and to avoid shell injection
    import subprocess
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout


def parse_usalign_chain_orders(output: str):
    """
    Parse USalign output for both Structure_1 and Structure_2 chain lists.
    Returns (chain_list_structure1, chain_list_structure2).
    """
    chain1 = None
    chain2 = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('Name of Structure_1:'):
            parts = line.split(':')
            clist = []
            for part in parts[2:]:
                token = part.strip()
                if token == '':
                    continue
                token0 = token.split()[0]
                last = token0.split(',')[-1]
                ch = re.sub(r'[^A-Za-z0-9]', '', last)
                if ch:
                    clist.append(ch)
            chain1 = clist
        elif line.startswith('Name of Structure_2:'):
            parts = line.split(':')
            clist = []
            for part in parts[2:]:
                token = part.strip()
                if token == '':
                    continue
                token0 = token.split()[0]
                last = token0.split(',')[-1]
                ch = re.sub(r'[^A-Za-z0-9]', '', last)
                if ch:
                    clist.append(ch)
            chain2 = clist
    if chain1 is None or chain2 is None:
        raise ValueError("Failed to parse chain orders from USalign output")
    return chain1, chain2


# ------------------------------------------------------------
# PDB writers (as in tm-score-permutechains.ipynb)
# ------------------------------------------------------------

def sanitize(xyz):
    MIN_COORD = -999.999
    MAX_COORD = 9999.999
    return min(max(xyz, MIN_COORD), MAX_COORD)


def write_target_line(atom_name, atom_serial, residue_name, chain_id, residue_num,
                      x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P') -> str:
    return f"ATOM  {atom_serial:>5d}  {atom_name:4s}{residue_name:>3s} {chain_id:1s}{residue_num:>4d}    {sanitize(x_coord):>8.3f}{sanitize(y_coord):>8.3f}{sanitize(z_coord):>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n"


def write2pdb(df: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    resolved_cnt = 0
    with open(target_path, 'w') as fh:
        for _, row in df.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                resolved_cnt += 1
                resid_num = int(row['resid'])
                fh.write(write_target_line("C1'", resid_num, row['resname'], 'A', resid_num, x, y, z, atom_type='C'))
    return resolved_cnt


def write2pdb_singlechain_native(df_native: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    df_sorted = df_native.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    resolved_cnt = 0
    with open(target_path, 'w') as fh:
        for _, row in df_sorted.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                resolved_cnt += 1
                resid_num = int(row['resid'])
                fh.write(write_target_line("C1'", resid_num, row['resname'], 'A', resid_num, x, y, z, atom_type='C'))
    return resolved_cnt


def write2pdb_multichain_from_solution(df_solution: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    df_sorted = df_solution.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int')

    chain_map = {}
    next_ord = ord('A')
    written = 0
    with open(target_path, 'w') as fh:
        for _, row in df_sorted.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if not (x > -1e6 and y > -1e6 and z > -1e6):
                continue
            chain_val = row['chain']
            copy_key = int(row['copy'])
            g = (str(chain_val), copy_key)
            if g not in chain_map:
                if next_ord <= ord('Z'):
                    ch = chr(next_ord)
                else:
                    ov = next_ord - ord('Z') - 1
                    if ov < 26:
                        ch = chr(ord('a') + ov)
                    else:
                        ch = chr(ord('0') + (ov - 26) % 10)
                chain_map[g] = ch
                next_ord += 1
            chain_id = chain_map[g]
            written += 1
            resid_num = int(row['resid'])
            fh.write(write_target_line("C1'", resid_num, row['resname'], chain_id, resid_num, x, y, z, atom_type='C'))
    return written


def write2pdb_multichain_from_groups(df_pred: pd.DataFrame, xyz_id: int, target_path: str, groups_list) -> (int, list):
    df_sorted = df_pred.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    if groups_list is None or len(groups_list) != len(df_sorted):
        raise ValueError("groups_list must be provided and match number of residues in predicted df")

    chain_map = {}
    next_ord = ord('A')
    chain_letters = []
    written = 0
    with open(target_path, 'w') as fh:
        for idx, row in df_sorted.iterrows():
            g = groups_list[idx]
            if isinstance(g, tuple):
                gkey = (str(g[0]), int(g[1]))
            else:
                gkey = (str(g), None)
            if gkey not in chain_map:
                if next_ord <= ord('Z'):
                    ch = chr(next_ord)
                else:
                    ov = next_ord - ord('Z') - 1
                    if ov < 26:
                        ch = chr(ord('a') + ov)
                    else:
                        ch = chr(ord('0') + (ov - 26) % 10)
                chain_map[gkey] = ch
                next_ord += 1
            chain_id = chain_map[gkey]
            chain_letters.append(chain_id)
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                written += 1
                resid_num = int(row['resid'])
                fh.write(write_target_line("C1'", resid_num, row['resname'], chain_id, resid_num, x, y, z, atom_type='C'))
    return written, chain_letters


def write2pdb_singlechain_permuted_pred(df_pred: pd.DataFrame, xyz_id: int, permuted_indices: list, target_path: str) -> int:
    df_sorted = df_pred.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    written = 0
    next_res = 1
    with open(target_path, 'w') as fh:
        for idx in permuted_indices:
            if idx < 0 or idx >= len(df_sorted):
                raise IndexError(f"permuted index {idx} out of range for predicted residues")
            row = df_sorted.iloc[idx]
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            out_resnum = next_res
            if x > -1e6 and y > -1e6 and z > -1e6:
                written += 1
                fh.write(write_target_line("C1'", out_resnum, row['resname'], 'A', out_resnum, x, y, z, atom_type='C'))
            next_res += 1
    return written


# ------------------------------------------------------------
# Target scoring (single target)
# ------------------------------------------------------------

def _score_one_prediction(pred_cnt, is_multicopy, group_predicted, group_native,
                          native_with_coords, tmpdir, usalign_bin, target_id):
    """Score a single prediction against all native frames. Returns best TM-score."""
    if not is_multicopy:
        predicted_pdb = os.path.join(tmpdir, f'predicted_{target_id}_{pred_cnt}.pdb')
        resolved_pred = write2pdb(group_predicted, pred_cnt, predicted_pdb)
        if resolved_pred <= 2:
            return 0.0

        scores = []
        for native_cnt in native_with_coords:
            native_pdb = os.path.join(tmpdir, f'native_{target_id}_{native_cnt}.pdb')
            out = run_usalign_raw(predicted_pdb, native_pdb, usalign_bin=usalign_bin, align_sequence=False, tmscore=1)
            s = parse_tmscore_output(out)
            scores.append(s)
        return max(scores)

    # multicopy
    gn_sorted = group_native.copy()
    gn_sorted['__resid_int'] = gn_sorted['resid'].astype(int)
    gn_sorted = gn_sorted.sort_values('__resid_int').reset_index(drop=True)
    groups_list = []
    for _, r in gn_sorted.iterrows():
        groups_list.append((r['chain'], int(r['copy'])))

    dfp_sorted = group_predicted.copy()
    dfp_sorted['__resid_int'] = dfp_sorted['resid'].astype(int)
    dfp_sorted = dfp_sorted.sort_values('__resid_int').reset_index(drop=True)
    if len(groups_list) != len(dfp_sorted):
        raise ValueError(f"groups_list length ({len(groups_list)}) != predicted residues ({len(dfp_sorted)}) for {target_id}")

    predicted_multi_pdb = os.path.join(tmpdir, f'pred_multi_{target_id}_{pred_cnt}.pdb')
    resolved_pred_multi, pred_chain_letters = write2pdb_multichain_from_groups(
        group_predicted, pred_cnt, predicted_multi_pdb, groups_list)
    if resolved_pred_multi == 0:
        return 0.0

    scores = []
    for native_cnt in native_with_coords:
        # Include pred_cnt in filenames to avoid collisions across parallel predictions
        native_multi_pdb = os.path.join(tmpdir, f'native_multi_{target_id}_{pred_cnt}_{native_cnt}.pdb')
        resolved_native_multi = write2pdb_multichain_from_solution(group_native, native_cnt, native_multi_pdb)
        if resolved_native_multi == 0:
            continue

        raw_out = run_usalign_raw(predicted_multi_pdb, native_multi_pdb,
                                  usalign_bin=usalign_bin, align_sequence=True, tmscore=0)
        chain1, chain2 = parse_usalign_chain_orders(raw_out)

        native_to_pred = {n_ch: p_ch for n_ch, p_ch in zip(chain2, chain1)}
        native_chain_order = sorted(native_to_pred.keys())
        pred_chain_order = [native_to_pred[n_ch] for n_ch in native_chain_order
                           if native_to_pred.get(n_ch) is not None]

        pred_positions_by_chain = {}
        for idx, ch in enumerate(pred_chain_letters):
            if ch is not None:
                pred_positions_by_chain.setdefault(ch, []).append(idx)

        pred_chain_order = [p for p in pred_chain_order if p in pred_positions_by_chain]

        permuted_indices = []
        for ch in pred_chain_order:
            permuted_indices.extend(pred_positions_by_chain[ch])
        for idx in range(len(pred_chain_letters)):
            if idx not in permuted_indices:
                permuted_indices.append(idx)

        pred_single_perm = os.path.join(tmpdir, f'pred_permuted_{target_id}_{pred_cnt}_{native_cnt}.pdb')
        written_pred_single = write2pdb_singlechain_permuted_pred(
            group_predicted, pred_cnt, permuted_indices, pred_single_perm)
        native_single = os.path.join(tmpdir, f'native_single_{target_id}_{pred_cnt}_{native_cnt}.pdb')
        written_native = write2pdb_singlechain_native(group_native, native_cnt, native_single)

        if written_pred_single <= 2 or written_native <= 2:
            raise ValueError(f"Insufficient residues after permutation for {target_id}, pred {pred_cnt}, native {native_cnt}")

        out = run_usalign_raw(pred_single_perm, native_single,
                              usalign_bin=usalign_bin, align_sequence=False, tmscore=1)
        scores.append(parse_tmscore_output(out))

    return max(scores)


def score_target(target_id: str, group_native: pd.DataFrame, group_predicted: pd.DataFrame, usalign_bin: str) -> float:
    has_chain_copy = ('chain' in group_native.columns) and ('copy' in group_native.columns)
    is_multicopy = has_chain_copy and (group_native['copy'].astype(float).max() > 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # precompute native models that have coords
        native_with_coords = []
        for native_cnt in range(1, 41):
            native_pdb = os.path.join(tmpdir, f'native_{target_id}_{native_cnt}.pdb')
            resolved_native = write2pdb(group_native, native_cnt, native_pdb)
            if resolved_native > 0:
                native_with_coords.append(native_cnt)
            else:
                if os.path.exists(native_pdb):
                    os.remove(native_pdb)

        if not native_with_coords:
            raise ValueError(f"No native models with coordinates for target {target_id}")

        with ThreadPoolExecutor(max_workers=5) as inner_ex:
            futures = [
                inner_ex.submit(_score_one_prediction, pred_cnt, is_multicopy,
                                group_predicted, group_native, native_with_coords,
                                tmpdir, usalign_bin, target_id)
                for pred_cnt in range(1, 6)
            ]
            best_per_pred = [f.result() for f in futures]

        return max(best_per_pred)


# ------------------------------------------------------------
# Main scoring (parallel across targets)
# ------------------------------------------------------------

def score_parallel(solution: pd.DataFrame, submission: pd.DataFrame, usalign_bin: str, workers: int, mode: str = 'thread') -> float:
    sol = solution.copy()
    sub = submission.copy()
    sol['target_id'] = sol['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))
    sub['target_id'] = sub['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))

    targets = list(sol['target_id'].unique())

    groups_native = {tid: sol[sol['target_id'] == tid] for tid in targets}
    groups_pred = {tid: sub[sub['target_id'] == tid] for tid in targets}

    per_target = {}
    if mode == 'process':
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(score_target, tid, groups_native[tid], groups_pred[tid], usalign_bin): tid for tid in targets}
            for fut in as_completed(futures):
                per_target[futures[fut]] = fut.result()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(score_target, tid, groups_native[tid], groups_pred[tid], usalign_bin): tid for tid in targets}
            for fut in as_completed(futures):
                per_target[futures[fut]] = fut.result()

    mean_tm = float(sum(per_target.values()) / len(per_target)) if per_target else 0.0
    return mean_tm, per_target


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multithreaded local TM-score validation (tm-score-permutechains compatible).')
    parser.add_argument('submission', type=str, help='Path to submission.csv')
    parser.add_argument('--validation', type=str, default=None, help='Path to validation_labels.csv')
    parser.add_argument('--usalign', type=str, default=None, help='Path to USalign binary')
    parser.add_argument('--workers', type=int, default=0, help='Number of worker threads (default: min(cpu_count, targets))')
    parser.add_argument('--mode', type=str, default='thread', choices=['thread', 'process'], help='Parallel mode: thread or process')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    validation_path = Path(args.validation) if args.validation else project_root.parent / 'stanford-rna-3d-folding-2' / 'validation_labels.csv'
    usalign_bin = args.usalign if args.usalign else str(Path(__file__).parent / 'USalign')

    if not Path(usalign_bin).exists():
        raise FileNotFoundError(f'USalign not found at {usalign_bin}. Compile with: g++ -O3 -ffast-math -o USalign USalign.cpp')

    submission = pd.read_csv(args.submission)
    validation = pd.read_csv(validation_path)

    cpu_threads = os.cpu_count() or 1
    targets = submission['ID'].str.rsplit('_', n=1).str[0].nunique()
    workers = args.workers if args.workers and args.workers > 0 else min(cpu_threads, targets)

    print(f"Detected CPU threads: {cpu_threads}")
    print(f"Targets: {targets} | Using workers: {workers}")

    mean_tm, per_target = score_parallel(validation, submission, usalign_bin, workers, mode=args.mode)
    for tid in sorted(per_target):
        print(f"TM:{tid}={per_target[tid]:.6f}")
    print(f"Mean TM-score: {mean_tm:.6f}")


if __name__ == '__main__':
    main()
