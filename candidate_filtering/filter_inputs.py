import numpy as np
import pickle
import os 
import sys
import argparse

sys.path.append("..")
from codegen.model.src.utils import submit_function_with_given_scripts, read_file_lines
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_dir", type=str, default='/scratch/yiqingxi')
    parser.add_argument("--result_dir", type=str, default='code_trans_output/new_g4g/codet5_base/java2python_bs4_8_epoch20_lr1e-4/transcoder_eval_ppl_multi_output_0.8')
    parser.add_argument("--id_file", type=str, default='CMTrans/data/transcoder_test_gfg/test.java-python.id')
    parser.add_argument("--func_file", type=str, default='CMTrans/data/transcoder_test_gfg/test.java-python.java')
    parser.add_argument("--lang", type=str, default='java')
    parser.add_argument("--subset_folder", type=str, default='func_subsets')
    parser.add_argument("--subset_idx", type=int, default=0)
    
    args = parser.parse_args()
    
    subset_idx = args.subset_idx
    subset_folder = args.subset_folder
    
    result_dir = args.result_dir
    tmp_dir = args.tmp_dir
    
    # input files
    src_path = args.func_file
    id_path = args.id_file
    test_input_file = result_dir + '/' + subset_folder + '/' + str(subset_idx) + '/test_inputs.pkl'
    
    # output files
    out_path = tmp_dir + '/' + subset_folder + '/' + str(subset_idx) + '/playground'
    test_case_file = result_dir + '/' + subset_folder + '/' + str(subset_idx) + '/test_cases.pkl'

    print('Loading results from', result_dir)
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    with open(test_input_file, 'rb') as fin:
        test_input_by_func = pickle.load(fin)
        problem_subset = set(test_input_by_func.keys())

    functions = read_file_lines(src_path)
    ids = read_file_lines(id_path)
    funcs = []
    func_ids = []
    for i in range(len(functions)):
        f, f_id = functions[i], ids[i].strip()
        if f_id not in problem_subset:
            continue
        funcs.append(f)
        func_ids.append(f_id)
    print(len(funcs), len(problem_subset))

    executor = ProcessPoolExecutor()
    test_cases_by_func = {}
    for i in range(len(funcs)):
        func, func_id = funcs[i], func_ids[i]
        existing_inputs = set()
        successful_flag = False
        for test_input_id in test_input_by_func[func_id]:
            input_scripts = test_input_by_func[func_id][test_input_id]['scripts'][args.lang]
            if test_input_by_func[func_id][test_input_id]['global_scripts'] is not None:
                input_global_scripts = test_input_by_func[func_id][test_input_id]['global_scripts'][args.lang]
            else:
                input_global_scripts = ''
                
            # print([i], test_input_id, func_id)
            job = executor.submit(
                        submit_function_with_given_scripts,
                        func,
                        input_scripts,
                        func_id,
                        args.lang,
                        out_path,
                        input_global_scripts,
                        '_'+str(test_input_id),
                    )
            
            result = job.result()
            
            if result is not None:
                pred, error_msg = result
                # print(result, '\n')

                # if pred not in ['error', 'timeout'] and 'Error' not in error_msg and 'error' not in error_msg:
                if error_msg == '':
                    if func_id not in test_cases_by_func:
                        test_cases_by_func[func_id] = {}
                        
                    test_cases_by_func[func_id][test_input_id] = (pred, test_input_by_func[func_id][test_input_id])
                    successful_flag = True
                elif pred == 'timeout':
                    break 
                elif error_msg is not None:
                    if args.lang == 'java':
                        if 'error: cannot find symbol' in error_msg:
                            # skip syntax errors
                            break 
                        
        if not successful_flag:
            print([i], test_input_id, func_id)

    print('test on', len(funcs) , '/', len(problem_subset), 'functions')
    print('has valid inputs for', len(test_cases_by_func), '/', len(funcs), 'functions')

    # save results to file 
    print('Saving successful results to file..')
    with open(test_case_file, 'wb') as fout:
        pickle.dump(test_cases_by_func, fout)
