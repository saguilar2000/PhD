#!/bin/bash

python3 parallel_process_limit.py 1 0 drift 0.0 && python3 parallel_process_limit.py 1 0 drift 0.5 && python3 parallel_process_limit.py 1 0 drift 1.0 && python3 parallel_process_limit.py 1 0 nodrift 0.0 && python3 parallel_process_limit.py 1 0 nodrift 0.5 && python3 parallel_process_limit.py 1 0 nodrift 1.0