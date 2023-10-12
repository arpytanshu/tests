
Just some code snippets to look back at. 

    `
    \  
    |-  
    |- README.md                    - this file
    |- localLLM_streamlit_app.py    - app for using LLMs locally.
    |- llm_finetuning/              - drivers to test GA, GC, LoRA on LLMs.
    |- llm_quantization/            - drivers to quantize llms.
    |- llm_evaluation/              - drivers to evaluate llms.
        |- evaluate_mmlm_hf.py      - evaluate hf models on mmlm benchmark.
    |- customForward/               - code to visualize dist over tokens across intermediate LLM layers.
    |- plots/                       - random plots
    |- docs/  
        |- cmake.md  
        |- cuda.md
        |- git.md  
        |- libtorch.md  
        |- markdown.md
    |- util_snippets/  
        |- ctypes/                  - using c/c++ code in python using ctypes.
        |- cuda/                    - getting started with cuda.
        |- LLMs/                    - model file from nanoGPT.
        |- pybind11/                - using c/c++ code in python using pybind11.
        |- torch_cpp/               - libtorch demo files.
        |- torch_cpp_extensions/    - c++/cuda extensions for torch using pybind11 and jit
    |- random/
        |- micrograd/               - AK's micrograd.
        |- colab_functions.py       - utility functions for colab
        |- efficient_training...py  - code from HF tutorial.
        |- plot_norms.py            - 
        |- plot_positional_emb..py  - 
        |- plot_vector_norms.py     - 
        
    `