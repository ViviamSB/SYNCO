import os
import shutil
import zipfile
from datetime import datetime

# //////////////////////////////////////////////////////////////////////////////////
def make_boolevard_input(
                        models_source,   # Directory with the b.net files
                        pipeline_main_source,   # Directory with the "perturbations", "modeloutputs" files
                        pipeline_supp_source,   # Directory with the ""drug_panel_df.csv" files
                        dest_dir,   # Directory where the BooLEVARD folder will be created
                        output_folder   # Final folder name where the zip files will be stored
                        ):
    print("Step 1: Copying .bnet model files...")
    _copy_bnet_models(models_source, dest_dir)
    
    print("Step 2: Copying pipeline files...")
    _copy_pipeline_files(pipeline_main_source, dest_dir)

    print("Step 3: Copying drug panel file...")
    _copy_pipeline_files(pipeline_supp_source, dest_dir, ["drug_panel_df.csv"])

    print("Step 4: Zipping cell line folders...")
    _zip_cell_line_folders(dest_dir, output_folder)

    print("BooLEVARD zip files generated succesfully")


def _copy_bnet_models(source_dir, dest_dir):
    # Loop through each cell line folder
    for cell_line in os.listdir(source_dir):
        cell_line_path = os.path.join(source_dir, cell_line)
        if not os.path.isdir(cell_line_path):
            continue  # skip files

        # Get the first dated folder inside the cell line folder
        subfolders = [f for f in os.listdir(cell_line_path) 
                    if os.path.isdir(os.path.join(cell_line_path, f))]
        if not subfolders:
            print(f"No dated folders in {cell_line_path}")
            continue
        date_folder = subfolders[0]  # assume only one

        models_path = os.path.join(cell_line_path, date_folder, "models")
        if not os.path.exists(models_path):
            print(f"No models folder in {models_path}")
            continue

        # Make sure the destination path exists
        target_path = os.path.join(dest_dir, cell_line, "Models")
        os.makedirs(target_path, exist_ok=True)

        # Copy only .bnet files
        for file in os.listdir(models_path):
            if file.endswith(".bnet"):
                src_file = os.path.join(models_path, file)
                dst_file = os.path.join(target_path, file)
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} → {dst_file}")


def _copy_pipeline_files(
        src_dir, 
        dest_dir,
        files_to_copy = ["perturbations", "modeloutputs"] # or ["drug_panel_df.csv"]
        ):

    for cell_line in os.listdir(dest_dir):
        cell_line_path = os.path.join(dest_dir, cell_line)        
        if not os.path.isdir(cell_line_path):
            continue

        # Make sure the src destination path exists
        dest_src_path = os.path.join(dest_dir, cell_line, "src")
        os.makedirs(dest_src_path, exist_ok=True)

        for filename in files_to_copy:
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dest_src_path, filename)

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} → {dst_file}")
            else:
                print(f"Missing source file: {src_file}")


def _zip_cell_line_folders(base_dir, output_dir):
    zip_output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(zip_output_dir, exist_ok=True)

    for name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, name)
        if not os.path.isdir(folder_path) or name == output_dir:
            continue  # skip files and the zip folder itself

        zip_path = os.path.join(zip_output_dir, f"{name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=folder_path)
                    zipf.write(file_path, arcname=arcname)

        print(f"Zipped {folder_path} → {zip_path}")


# //////////////////////////////////////////////////////////////////////////////////
def extract_boolevard_results(zip_dir, out_results, tmp_dir=None):
    if tmp_dir is None:
        tmp_dir = os.path.join(zip_dir, "tmp_unzip")
    os.makedirs(tmp_dir, exist_ok=True)
    
    missing_report = {}
    expected_files = []

    for zip_file in os.listdir(zip_dir):
        if not zip_file.endswith(".zip"):
            continue

        cell_line = os.path.splitext(zip_file)[0]
        zip_path = os.path.join(zip_dir, zip_file)
        extract_path = os.path.join(tmp_dir, cell_line)

        unzip_cell_line_folder(zip_path, extract_path)
        copy_BL_results(cell_line, extract_path, out_results,
                        expected_files=expected_files,
                        missing_report=missing_report)

    # Final cleanup
    shutil.rmtree(tmp_dir)
    print("Temporary folder cleaned up.\n")
    print("All results extracted.")

    if missing_report:
        print("\nMissing Report:")
        for cell_line, issue in missing_report.items():
            print(f"  - {cell_line}: {issue}")

def unzip_cell_line_folder(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)
    print(f"Unzipped {os.path.basename(zip_path)} to {extract_to}")

def copy_BL_results(cell_line, extract_path, out_results, expected_files=None, missing_report=None):

    results_path = os.path.join(extract_path, "Results")
    if not os.path.exists(results_path):
        print(f"No Results folder for {cell_line}")
        if missing_report is not None:
            missing_report[cell_line] = "Missing Results folder"
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(out_results, cell_line, f"{cell_line}_{timestamp}")
    os.makedirs(final_path, exist_ok=True)

    # Copy files
    for item in os.listdir(results_path):
        src = os.path.join(results_path, item)
        dst = os.path.join(final_path, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    # Check for expected files
    expected_files = [
        f"PathCounts_{cell_line}.tsv",
        f"SynergyExcess_{cell_line}.tsv"
    ]

    missing = [f for f in expected_files if not os.path.exists(os.path.join(results_path, f))]

    if missing and missing_report is not None:
        missing_report[cell_line] = f"Missing files: {', '.join(missing)}"
    else:
        print(f"Copied results to {final_path}")

# //////////////////////////////////////////////////////////////////////////////////
## TESTING ##

# Working directory
base_dir = 'C:\\Users\\viviamsb\\OneDrive - NTNU\\PhD Folder\\Pipeline\\DrugLogics_pipeline_modules\\run_all\\run_vis_cell_fate\\hsa\\20250718'

# EXTRACT RESULTS FROM ZIPS
zip_dir_test = base_dir + '\\BooLEVARD_zips\\BL_out'
out_results_test = base_dir + '\\BooLEVARD_zips\\BL_results'
# extract_boolevard_results(zip_dir_test, out_results_test, tmp_dir=None)


# //////////////////////////////////////////////////////////////////////////////////
# SET VARIABLES
zip_folders = "CL_zips"  # Folder where the zip files will be stored
results_folders = "CL_results"  # Folder where the results will be stored

# CALL FUNCTIONS
# make_boolevard_input(
#     models_source="models_out",  # Directory with the b.net files
#     pipeline_main_source="pipeline_files",  # Directory with the "perturbations", "modeloutputs" files
#     pipeline_supp_source="supplementary_files",  # Directory with the "drug_panel_df.csv" files
#     dest_dir="BooLEVARD_zips",  # Directory where the BooLEVARD folder will be created
#     output_folder=zip_folders  # Final folder name where the zip files will be stored
# )

# extract_boolevard_results(
#     zip_dir=zip_folders,  # Directory with the zip files
#     out_results=results_folders,  # Directory where the results will be stored
#     tmp_dir=None  # Temporary directory for unzipping
# )