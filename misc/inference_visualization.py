import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def parser(src_dir: str) -> pd.DataFrame:
    """
    A helper function to read/parse all json report files and combine the metrics for visualization.

    : param src_dir: The directory that contains the json reports for each model format.
    : return: A pandas dataframe that combines all metrics for each model format i.e., PyTorch, ONNX, OpenVINO.
    """
    column_names = ['framework', 'ao', 'sr_50', 'sr_75', 'speed_fps', 'succ_curve']
    df = pd.DataFrame(columns=column_names)
    paths = ["SMAT_pytorch", "SMAT_onnx", "SMAT_openvino", "SMAT_openvino*"]

    for path in paths:
        path_to_file = os.path.join(src_dir, path + '.json')
        with open(path_to_file, 'r') as json_file:
            data =json.load(json_file)
            for key in data.keys():
                # Create a new row for each iteration
                new_row = {
                    'framework': key,
                    'ao': data[key]['ao'],
                    'sr_50': data[key]['sr_50'],
                    'sr_75': data[key]['sr_75'],
                    'speed_fps': data[key]['speed_fps'],
                    'succ_curve': [data[key]['succ_curve']] # Wrap the list into a list
                }
                # Append the new row to the DataFrame
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    print(df.head())

    # Rounding floating point values to 4 decimal places 
    df = df.round(4)
    # Save the DataFrame as a CSV file
    df.to_csv(src_dir + '/results.csv', index=False)    
    return df

def visualization(src_dir: str, df: pd.DataFrame) -> None:
    """
    A helper function that plots and saves the figures for the metrics reported using the pandas.

    : param: Directory to save the figure generated in.
    :param: Pandas framework which contains all metrics. 
    """
    sns.set_theme()
    fig, ax = plt.subplots(figsize =(8, 5))
    sns.barplot(df, x=df['framework'], y=df['speed_fps'])
    ax.bar_label(ax.containers[0], fontsize=10)
    fig.savefig(src_dir + '/speed_fps.png')


if __name__ == '__main__':
    src_dir = "/home/hassan/SMAT_2/SMAT/output/test/tracking_results/mobilevitv2_track/results"
    df = parser(src_dir)
    visualization(src_dir, df)
