Large Language Models (LLMs), such as Llama
3.1 8B, offer state-of-the-art natural language
processing capabilities but present significant
deployment challenges on resource-constrained
hardware. Standard implementations of Llama
3.1 8B require approximately 16GB of RAM in
half-precision (fp16) and up to 32GB in single
precision (fp32) to load and perform inference.
This substantial memory footprint renders deploy
ment on common edge devices, such as the Rasp
berry Pi 4, impossible due to their strict hard
ware limitations. In this work, we address the
challenge of running high-performance LLMs on
edge hardware by compressing Llama 3.1 8B to fit
within the 8GB RAM capacity of a Raspberry Pi
4. By employing model compression techniques
and utilizing the llama.cpp framework for op
timized C++ inference, we successfully reduced
the model’s memory requirements
