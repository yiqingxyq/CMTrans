import numpy as np
import pickle
import os 
import sys
import argparse

sys.path.append("..")
from codegen.model.src.utils import submit_function_with_given_scripts, read_file_lines
from concurrent.futures import ProcessPoolExecutor

OVERFLOW_BOUND=10000000

def compare_answer(gt, pred):
    gt, pred = gt.strip(), pred.strip()
    # float 
    try:
        gt_float = float(gt)
        pred_float = float(pred)
        if gt_float > OVERFLOW_BOUND or gt_float < -OVERFLOW_BOUND:
            return True # overflow, pass this case

        if round(gt_float, 2) == round(pred_float, 2):
            return True
        else:
            return False 
    except:
        pass 

    # bool 
    if gt in ['false', 'true'] or pred in ['False', 'True']:
        if gt.lower() == pred.lower():
            return True 
        elif gt == '1' and pred == 'True' or gt == '0' and pred == 'False':
            return True
        else:
            return False 

    # String 
    gt_str = gt.replace('\n','').replace('\t','').replace(' ','')
    pred_str = pred.replace('\n','').replace('\t','').replace(' ','')
    if gt_str == pred_str:
        return True 
    else:
        return False

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp_dir", type=str, default='/scratch/yiqingxi')
    parser.add_argument("--result_dir", type=str, default='/projects/tir6/general/yiqingxi/avatar_output/new_g4g/codet5_base/java2python_bs4_8_epoch20_lr1e-4/transcoder_eval_ppl_multi_output_0.8')
    parser.add_argument("--id_file", type=str, default='/home/yiqingxi/AVATAR-2/data/transcoder_test_gfg/test.java-python.id')
    parser.add_argument("--lang", type=str, default='java')
    parser.add_argument("--candidate_start", type=int, default=0)
    parser.add_argument("--candidate_end", type=int, default=20)
    parser.add_argument("--test_case_num", type=int, default=10)
    parser.add_argument("--subset_folder", type=str, default='func_subsets')
    parser.add_argument("--subset_idx", type=int, default=0)
    
    args = parser.parse_args()
    
    tmp_dir = args.tmp_dir
    result_dir = args.result_dir
    candidate_start = args.candidate_start
    candidate_end = args.candidate_end
    test_case_num = args.test_case_num
    subset_idx = args.subset_idx
    subset_folder = args.subset_folder
    

    id_path = args.id_file
    
    # inputs
    CANDIDATE_FILE_PREFIX = result_dir + '/test.output'
    test_case_file = result_dir + '/' + subset_folder + '/' + str(subset_idx) + '/test_cases.pkl'
    
    # outputs
    selected_cand_file = result_dir + '/' + subset_folder + '/' + str(subset_idx) + \
                        '/selected_cands'+str(candidate_start) + '_' + str(candidate_end) +'.pkl'
    result_file = result_dir + '/' + subset_folder + '/' + str(subset_idx) + \
                        '/func_id2cand2results'+str(candidate_start) + '_' + str(candidate_end) +'.pkl'
    out_path_prefix = tmp_dir + '/' + subset_folder + '/' + str(subset_idx) + '/playground_candidate'
    
    print('Filtering candidates in', result_dir)

    test_cases_by_func = pickle.load(open(test_case_file, 'rb'))
    for f in test_cases_by_func:
        # Keep <= test_case_num cases for each function 
        if len(test_cases_by_func[f]) > test_case_num:
            remove_subset = np.random.choice(list(test_cases_by_func[f].keys()), len(test_cases_by_func[f]) - test_case_num, replace=False)
            for k in remove_subset:
                del test_cases_by_func[f][k]
        
    print('Avg num of test inputs:', np.mean([len(test_cases_by_func[x]) for x in test_cases_by_func]))

    funcs = {}
    func_ids = []
    ids = read_file_lines(id_path)
    for i in range(len(ids)):
        f_id = ids[i].strip()
        if f_id not in test_cases_by_func:
            continue 
        func_ids.append(f_id)

    for cid in range(candidate_start, candidate_end):
        result_path = CANDIDATE_FILE_PREFIX + str(cid)
        out_path = out_path_prefix + str(cid)

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        functions = read_file_lines(result_path)
        funcs[cid] = []
        for i in range(len(functions)):
            f, f_id = functions[i], ids[i].strip()
            if f_id not in test_cases_by_func:
                continue
            funcs[cid].append(f)

    executor = ProcessPoolExecutor()
    results = {}
    func_id2selected_cand = {}
    for i in range(len(func_ids)):
        print('Processing Function id:', i)
        func_id = func_ids[i]
        results[func_id] = {}
        func2res = {}
        for cid in range(candidate_start, candidate_end):
            result_path = CANDIDATE_FILE_PREFIX + str(cid)
            out_path = out_path_prefix + str(cid)
            func, func_id = funcs[cid][i], func_ids[i]

            if func in func2res:
                # exclude repeated functions
                results[func_id][cid] = func2res[func]
                # print('Repeated func. Skipping testing')
                continue

            results[func_id][cid] = {}
            successful_flag = True
            for test_input_id in test_cases_by_func[func_id]:
                # print([i, test_input_id, cid])
                gt, inputs = test_cases_by_func[func_id][test_input_id]
                input_scripts = inputs['scripts'][args.lang]
                if inputs['global_scripts'] is not None:
                    input_global_scripts = inputs['global_scripts'][args.lang]
                else:
                    input_global_scripts = ''
                
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

                    if pred not in ['error', 'timeout'] and 'Error' not in error_msg and 'error' not in error_msg:
                        res = compare_answer(gt, pred)
                        if res:
                            # print('[CORRECT]', func_id)
                            # print('GT:', gt, )
                            # print('OURS:', pred, '\n')
                            results[func_id][cid][test_input_id] = ('success', None)
                        else:
                            successful_flag = False
                            # print(func_id)
                            # print('GT:', gt, )
                            # print('OURS:', pred, '\n')
                            results[func_id][cid][test_input_id] = ('failure', error_msg)
                    else:
                        successful_flag = False 
                        # print(func_id)
                        # print('GT:', (gt, ), )
                        # print('OURS:', result, '\n')

                        if pred in ['timeout']:
                            # skip all inputs 
                            for j in test_cases_by_func[func_id]:
                                if j not in results[func_id][cid]:
                                    results[func_id][cid][j] = ('timeout', error_msg)
                            print('Timeout. Skip remaining test cases')
                            func2res[func] = results[func_id][cid]
                            break
                        else:
                            results[func_id][cid][test_input_id] = ('error', error_msg)
                else:
                    successful_flag = False
                    # print(func_id)
                    # print('GT:', (gt, ), )
                    # print('OURS:', result, '\n')
                    results[func_id][cid][test_input_id] = ('no_outputs', None)

            if successful_flag:
                if func_id not in func_id2selected_cand:
                    func_id2selected_cand[func_id] = []
                func_id2selected_cand[func_id].append(cid)

            func2res[func] = results[func_id][cid]
                

    print('test on', len(test_cases_by_func), '/', len(funcs[cid]) , 'functions')
    print('Successfully passed', len(func_id2selected_cand), '/', len(funcs[cid]), 'functions' )

    print('Saving to file..')
    with open(selected_cand_file, 'wb') as fout:
        pickle.dump(func_id2selected_cand, fout)
        
    with open(result_file, 'wb') as fout:
        pickle.dump(results, fout)
