import numpy as np
import pickle
import os 
import sys
import argparse
from tqdm import tqdm

sys.path.append("..")
from codegen.model.src.utils import generate_inputs, read_file_lines

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default='code_trans_output/new_g4g/codet5_base/java2python_bs4_8_epoch20_lr1e-4/transcoder_eval_ppl_multi_output_0.8')
    parser.add_argument("--id_file", type=str, default='CMTrans/data/transcoder_test_gfg/test.java-python.id')
    parser.add_argument("--func_file", type=str, default='CMTrans/data/transcoder_test_gfg/test.java-python.java')
    parser.add_argument("--lang", type=str, default='java')
    parser.add_argument("--test_case_num", type=int, default=10)
    parser.add_argument("--subset_folder", type=str, default='func_subsets')
    parser.add_argument("--subset_idx", type=int, default=0)
    
    args = parser.parse_args()
    
    test_case_num = args.test_case_num
    subset_idx = args.subset_idx
    subset_folder = args.subset_folder
    
    result_dir = args.result_dir
    
    # input files
    src_path = args.func_file
    id_path = args.id_file
    subset_file = result_dir + '/' + subset_folder + '/' + str(subset_idx) + '/problem_subset.pkl'
    
    # output files
    test_input_file = result_dir + '/' + subset_folder + '/' + str(subset_idx) + '/test_inputs.pkl'

    print('Loading results from', result_dir)

    if os.path.exists(subset_file):
        with open(subset_file, 'rb') as fin:
            problem_subset = pickle.load(fin)
    else:
        problem_subset = set()

    functions = read_file_lines(src_path)
    ids = read_file_lines(id_path)
    funcs = []
    func_ids = []
    for i in range(len(functions)):
        f, f_id = functions[i], ids[i].strip()
        if len(problem_subset)>0 and f_id not in problem_subset:
            continue
        funcs.append(f)
        func_ids.append(f_id)
    print(len(funcs), len(problem_subset))

    results = {}
    test_input_by_func = {}
    for i in tqdm(range(len(funcs))):
        func, func_id = funcs[i], func_ids[i]
        results[func_id] = []
        existing_inputs = set()
        for test_input_id in range(test_case_num):
            seed = np.random.randint(1000)
            global_out, out= generate_inputs(func, args.lang, seed)
            
            if out is not None:
                java_scripts, python_scripts, cpp_scripts, values = out

                if func_id not in test_input_by_func:
                    test_input_by_func[func_id] = {}
                
                if str(python_scripts) not in existing_inputs:
                    test_input_by_func[func_id][test_input_id] = {}
                    test_input_by_func[func_id][test_input_id]['scripts'] = {'java': java_scripts, 'python': python_scripts, 'cpp': cpp_scripts, 'values': values}
                    if global_out is not None:
                        global_java_scripts, global_python_scripts, global_cpp_scripts, global_values = global_out
                        test_input_by_func[func_id][test_input_id]['global_scripts'] = {'java': global_java_scripts, 'python': global_python_scripts, 'cpp': global_cpp_scripts, 'values': global_values}
                    else:
                        test_input_by_func[func_id][test_input_id]['global_scripts'] = None
                    
                    existing_inputs.add( str(python_scripts) )

    print('test on', len(funcs) , '/', len(problem_subset), 'functions')
    print('generate inputs for', len(test_input_by_func), '/', len(funcs), 'functions')

    print('Saving test inputs to file..')
    with open(test_input_file, 'wb') as fout:
        pickle.dump(test_input_by_func, fout)
