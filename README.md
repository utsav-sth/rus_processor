#rus_processor

rus_processor/
├── main.py
├── utils/
│   ├── reader.py
│   ├── writer.py
│   ├── gpu_kernels.py
│   ├── cleaner.py
├── kernels/
│   └── hit_filter.cu
├── config/
│   └── settings.yaml
├── README.md

This is a modular CUDA-accelerated Python pipeline for processing ROOT Universal Structure (RUS) files.
It reads ROOT using 'uproot', processes data using 'pycuda' and custon CUDA kernels, and writes the output as a cleaned/processed RUS ROOT file.

#Requirements
- Python 3.+
- CUDA-compatible GPU and Toolkit
- Python packages: numpy, uproot, pycuda

#Dependencies Installation:
pip install numpy uproot pycuda

#main.py
The main.py script serves as the central controller for the workflow that filters hits from a RUS-format ROOT file. It begins by loading configuration parameters specifically the input file path, output file path, and a drift distance threshold from a YAML file. Using the read_rus_file function, it loads event-level and hit-level data into awkward arrays, which are designed to handle jagged (variable-length) arrays efficiently. It removes the turnID field from the event variables if present, then flattens the hit-level driftDistance values and sends them to a GPU-based CUDA filtering function (run_hit_filtering) that returns a mask indicating which hits pass the threshold. This flat mask is reshaped into jagged form to align with the original event structure and is then applied across all hit-level branches to discard unwanted hits. Events with no remaining hits are entirely removed from both the hit and event data. After filtering, it calls a cleaning function to sanitize or remove unnecessary hit-level fields and finally writes the processed, filtered data to an output ROOT file using the write_rus_file function. The script also prints the final count of events and hits that remain after filtering.

#settings.yaml
The settings.yaml file provides the configuration parameters used by main.py. Here we can control the filtering workflow, specify the path to the input ROOT file (input.rus.root) and the output/processed ROOT file. Any standard or defined threshold values or constants used throughout the workflow can be assigned here.

#reader.py
The reader.py script defines the read_rus_file function, which reads and parses a RUS-format ROOT file using the uproot library. It extracts event-level variables into a dictionary (event_vars), separating scalar values like runID, eventID, etc., and fixed-length arrays such as fpgaTrigger, storing them as NumPy arrays, and also collects hit-level (jagged) variables—like hitID, detectorID, driftDistance, and tdcTime into a second dictionary (hit_vars) using Awkward Arrays. This will preserve their per-event variable-length structure. This will ensure that the hit data remains linked to its respective event information. Missing branches in the tree are skipped with warnings.

#writer.py
The writer.py script uses the write_rus_file function, which handles writing the processed and cleaned data back into a new ROOT file. The function merges the event-level variables (event_vars) and hit-level jagged arrays (hit_vars) into a single dictionary (all_vars), and then writes all of them into a TTree named "tree" within the output ROOT file. This keeps the output file structure consistent with the input.

#gpu_kernels.py
The gpu_kernels.py is pycuda script to use the actual CUDA kernel. Currently, a simple CUDA kernel named "hit_filter.cu" is being used. The idea is to flatten and convert the input jagged drift_distance array (from Awkward) into a contiguous NumPy array of type float64. It makes an empty output array (mask_output) of type int32 to hold the binary mask results (1 for keep, 0 for discard). It also allocates GPU memory for both the input drift distances and output mask, and then transfers the data to the GPU. It loads and compiles a CUDA kernel and the kernel function from the CUDA file. This kernel is launched with a block size of 256 threads and an appropriately sized grid based on the number of hits. This can be modified later. After the kernel execution, the output from the GPU is copied back.

#hit_filter.cu
The hit_filter.cu file is an exmaple of a simple CUDA kernel function. It simply performs element-wise filtering of drift distances. It takes an input array of drift values, an output mask array, the number of elements, and a drift_thresh value. Each thread computes a global index i and checks whether it's within bounds, and performs the kernel function, and give the output.

#cleaner.py
This is optional, but may be needed depending upon the processing. The cleaner.py processes the hit_vars dictionary to clean and prepare it for output.
