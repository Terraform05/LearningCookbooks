import os
import time
import subprocess
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend to avoid displaying plots
import matplotlib.pyplot as plt

# Define a function to execute each python file and measure its execution time
def time_execution(file_name):
    start_time = time.time()
    subprocess.run(["python", file_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    plt.close('all') # Close all open plots to suppress any displays
    end_time = time.time()
    return end_time - start_time

# Set the directory path to the desired folder
dir_path = "/Users/alexj/Code/LearningCookbooks/MachineLearning"

# Get a list of all files in the directory except the current Python file
file_names = [f for f in os.listdir(dir_path) if f.endswith('.py')]
file_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))


# Open the output file and write the Markdown table header
with open('/Users/alexj/Code/LearningCookbooks/timing/timing.md', 'w') as output_file:
    output_file.write('| File Name | Execution Time |\n')
    output_file.write('| --- | --- |\n')

    # Loop over each file name, execute it and write the output to the file
    for file_name in file_names:
        full_path = os.path.join(dir_path, file_name)
        execution_time = time_execution(full_path)
        output_file.write(f'| {file_name} | {execution_time:.2f} seconds |\n')
        print("\n\n\n","="*50,"\n","="*50,"\n",f"{file_name} completed.","\n","="*50,"\n","="*50,"")
