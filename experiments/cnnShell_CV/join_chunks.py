import pandas as pd
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
chunk_dir = os.path.join(script_dir, "outputs/updated")

use_temp = False
shell = "temp" if use_temp else "flux"

type_chunks = [
    f"amplitudes_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"amplitudes_perc_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"detections_binary_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"detections_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"periods_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"phases_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"detections_highest_peak_cnnshellCV5_9_{shell}_act_mask_res_chunk",

    # --- MCDO variance maps ---
    f"variance_rv_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"variance_ds_cnnshellCV5_9_{shell}_act_mask_res_chunk",

    # --- Residual maps (mean abs, and median abs) ---
    f"residuals_rv_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"residuals_ds_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"residuals_med_rv_cnnshellCV5_9_{shell}_act_mask_res_chunk",
    f"residuals_med_ds_cnnshellCV5_9_{shell}_act_mask_res_chunk",
]

output_names = [
    f"{shell}_amplitudes",
    f"{shell}_amplitudes_perc",
    f"{shell}_detections_binary",
    f"{shell}_detections",
    f"{shell}_periods",
    f"{shell}_phases",
    f"{shell}_detections_highest_peak",
    # --- NEW outputs ---
    f"{shell}_variance_rv",
    f"{shell}_variance_ds",
    f"{shell}_residuals_rv",
    f"{shell}_residuals_ds",
    f"{shell}_residuals_med_rv",
    f"{shell}_residuals_med_ds",
]

if len(type_chunks) != len(output_names):
    raise ValueError("type_chunks and output_names must have the same length.")

for prefix, outname in zip(type_chunks, output_names):
    file_list = sorted(glob.glob(os.path.join(chunk_dir, f"{prefix}*.csv")))
    if not file_list:
        raise FileNotFoundError(f"No files matched: {prefix}*.csv in {chunk_dir}")

    all_dfs = []
    for file in file_list:
        df = pd.read_csv(file, index_col=0)

        # Ensure numeric period index for proper sorting
        try:
            df.index = df.index.astype(float)
        except Exception:
            pass

        df.index.name = "Period"
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, axis=0)
    combined_df.sort_index(inplace=True)

    outpath = os.path.join(script_dir, f"detection_matrix_CV_{outname}_cnn.csv")
    combined_df.to_csv(outpath)
    print(f"Saved combined CSV for {outname}: {outpath}")
