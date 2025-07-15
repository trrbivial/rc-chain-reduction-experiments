import os
import json
import subprocess
import re


def run_ngspice(file_path, root):
    """运行 ngspice 并提取 Matrix factor time 和 Matrix solve time 数据"""
    temp_output = file_path + ".log"

    with open(temp_output, "w") as temp_file:
        process = subprocess.Popen(["ngspice", file_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   text=True,
                                   cwd=root)
        for line in process.stdout:
            print(line, end="")  # 同时输出到屏幕
            temp_file.write(line)


def extract_times(file_path):
    temp_output = file_path + ".log"
    times = {}
    with open(temp_output, "r") as temp_file:
        for line in temp_file:
            factor_match = re.search(r"Matrix factor time = ([\d\.]+)", line)
            solve_match = re.search(r"Matrix solve time = ([\d\.]+)", line)
            if factor_match:
                times["Matrix factor time"] = float(factor_match.group(1))
            if solve_match:
                times["Matrix solve time"] = float(solve_match.group(1))

            factor_complex_match = re.search(
                r"FactorComplexMatrix_time = ([\d\.]+)", line)
            count_markowitz_match = re.search(
                r"CountMarkowitz_time = ([\d\.]+)", line)
            markowitz_products_match = re.search(
                r"MarkowitzProducts_time = ([\d\.]+)", line)
            search_for_pivot_match = re.search(
                r"SearchForPivot_time = ([\d\.]+)", line)
            search_for_singleton_match = re.search(
                r"SearchForSingleton_time = ([\d\.]+)", line)
            quickly_search_diagonal_match = re.search(
                r"QuicklySearchDiagonal_time = ([\d\.]+)", line)
            search_diagonal_match = re.search(
                r"SearchDiagonal_time = ([\d\.]+)", line)
            search_entire_matrix_match = re.search(
                r"SearchEntireMatrix_time = ([\d\.]+)", line)
            find_largest_in_col_match = re.search(
                r"FindLargestInCol_time = ([\d\.]+)", line)
            find_biggest_in_col_exclude_match = re.search(
                r"FindBiggestInColExclude_time = ([\d\.]+)", line)
            exchange_rows_and_cols_match = re.search(
                r"ExchangeRowsAndCols_time = ([\d\.]+)", line)
            exchange_col_elements_match = re.search(
                r"ExchangeColElements_time = ([\d\.]+)", line)
            exchange_row_elements_match = re.search(
                r"ExchangeRowElements_time = ([\d\.]+)", line)
            real_row_col_elimination_match = re.search(
                r"RealRowColElimination_time = ([\d\.]+)", line)
            complex_row_col_elimination_match = re.search(
                r"ComplexRowColElimination_time = ([\d\.]+)", line)
            update_markowitz_numbers_match = re.search(
                r"UpdateMarkowitzNumbers_time = ([\d\.]+)", line)
            create_fillin_match = re.search(r"CreateFillin_time = ([\d\.]+)",
                                            line)
            matrix_is_singular_match = re.search(
                r"MatrixIsSingular_time = ([\d\.]+)", line)
            zero_pivot_match = re.search(r"ZeroPivot_time = ([\d\.]+)", line)
            sp_order_and_factor_match = re.search(
                r"spOrderAndFactor_time = ([\d\.]+)", line)
            sp_factor_match = re.search(r"spFactor_time = ([\d\.]+)", line)
            sp_partition_match = re.search(r"spPartition_time = ([\d\.]+)",
                                           line)
            spc_create_internal_vectors_match = re.search(
                r"spcCreateInternalVectors_time = ([\d\.]+)", line)

            if factor_complex_match:
                times["FactorComplexMatrix time"] = float(
                    factor_complex_match.group(1))
            if count_markowitz_match:
                times["CountMarkowitz time"] = float(
                    count_markowitz_match.group(1))
            if markowitz_products_match:
                times["MarkowitzProducts time"] = float(
                    markowitz_products_match.group(1))
            if search_for_pivot_match:
                times["SearchForPivot time"] = float(
                    search_for_pivot_match.group(1))
            if search_for_singleton_match:
                times["SearchForSingleton time"] = float(
                    search_for_singleton_match.group(1))
            if quickly_search_diagonal_match:
                times["QuicklySearchDiagonal time"] = float(
                    quickly_search_diagonal_match.group(1))
            if search_diagonal_match:
                times["SearchDiagonal time"] = float(
                    search_diagonal_match.group(1))
            if search_entire_matrix_match:
                times["SearchEntireMatrix time"] = float(
                    search_entire_matrix_match.group(1))
            if find_largest_in_col_match:
                times["FindLargestInCol time"] = float(
                    find_largest_in_col_match.group(1))
            if find_biggest_in_col_exclude_match:
                times["FindBiggestInColExclude time"] = float(
                    find_biggest_in_col_exclude_match.group(1))
            if exchange_rows_and_cols_match:
                times["ExchangeRowsAndCols time"] = float(
                    exchange_rows_and_cols_match.group(1))
            if exchange_col_elements_match:
                times["ExchangeColElements time"] = float(
                    exchange_col_elements_match.group(1))
            if exchange_row_elements_match:
                times["ExchangeRowElements time"] = float(
                    exchange_row_elements_match.group(1))
            if real_row_col_elimination_match:
                times["RealRowColElimination time"] = float(
                    real_row_col_elimination_match.group(1))
            if complex_row_col_elimination_match:
                times["ComplexRowColElimination time"] = float(
                    complex_row_col_elimination_match.group(1))
            if update_markowitz_numbers_match:
                times["UpdateMarkowitzNumbers time"] = float(
                    update_markowitz_numbers_match.group(1))
            if create_fillin_match:
                times["CreateFillin time"] = float(
                    create_fillin_match.group(1))
            if matrix_is_singular_match:
                times["MatrixIsSingular time"] = float(
                    matrix_is_singular_match.group(1))
            if zero_pivot_match:
                times["ZeroPivot time"] = float(zero_pivot_match.group(1))
            if sp_order_and_factor_match:
                times["spOrderAndFactor time"] = float(
                    sp_order_and_factor_match.group(1))
            if sp_factor_match:
                times["spFactor time"] = float(sp_factor_match.group(1))
            if sp_partition_match:
                times["spPartition time"] = float(sp_partition_match.group(1))
            if spc_create_internal_vectors_match:
                times["spcCreateInternalVectors time"] = float(
                    spc_create_internal_vectors_match.group(1))

    return times if times else None


def main():
    """遍历目录，查找文件并处理"""
    results = {}

    for root, _, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith(("_ann.net", "c7552.net")):
                file_path = os.path.join(root, file)
                run_ngspice(file_path, root)
                times = extract_times(file_path)
                if times:
                    results[file_path] = times

    output_json = os.path.join(os.getcwd(), "ngspice_times.json")
    with open(output_json, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"运行完成，时间数据已保存至 {output_json}")


if __name__ == "__main__":
    main()
