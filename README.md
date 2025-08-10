# Beyond Execution: Uncertainty Estimation and Abstention in LLM-Based Code Generation

Currently, we support code generation tasks using HumanEval, HumanEval+, MBPP, MBPP+. The code is supported to any LLM supported in HuggingFace. 

### We explain how to run our code in the below: 

## To run uncertainty metrics on generated tokens:

1. Go to ./CodeGen/provider/hf.py
 2. Uncomment line 104

## To run uncertainty metrics on generated tokens:

 1. Go to ./CodeGen/provider/hf.py
2. Uncomment line 106

## Running metrics after uncommenting line 104 or 106 from the CodeGen folder run the follwing command, 

python ./evaluate.py \
    --model "(model_name)" \
    --dataset "(dataset)" \
    --backend hf \
    --greedy \

"(model_name)" can be any HF model. E.g.: ./LLMs/CodeLlama_7b_Instruct_hf
"(dataset)" can be either mbpp or humaneval

## Running evaluation: 

Go to file: Change the path according to yours in CodeGen/compute_metrics.py (lines 15 and 16) and CodeGen/compute_abstaintion.py (lines 14 and 15)
Then run: 
python compute_metrics.py
python compute_abstaintion.py 

## Note that this directory is supported to any code LLM used in HF.





