from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib.pyplot as plt
from processing.csv_processing import plot, plot_with_symmetry_lines, first
import io
from copy import deepcopy

app = Flask(__name__)

def read_csv_(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    unique_combinations = np.unique(np_path_XYs[:, :2], axis=0)  # Get unique combinations of path_id and segment_id

    for path_id, segment_id in unique_combinations:
        npXYs = np_path_XYs[(np_path_XYs[:, 0] == path_id) & (np_path_XYs[:, 1] == segment_id)][:, 2:]  # Select all points for the path_id and segment_id, ignore the first two columns
        path_XYs.append(npXYs)

    return path_XYs

def plot_main(paths_XYs, outputdata, shape_count):
    fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=(16, 8))
    
    # Plot the three subplots
    plot(paths_XYs, 'Original Data', axs[0])
    plot(outputdata, 'Regularized Data', axs[1])
    plot_with_symmetry_lines(outputdata, 'Processed Data with Symmetry Lines', axs[2])

    # Add shape_count information as text on the last subplot (axs[2])
    text_str = '\n'.join([f'{key}: {value}' for key, value in shape_count.items()])
    axs[2].text(0.05, 0.95, text_str, transform=axs[2].transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    return fig

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            paths_XYs = read_csv_(file)
            inputdata = deepcopy(paths_XYs)
            outputdata, shape_count = first(inputdata)
            
            fig = plot_main(paths_XYs, outputdata, shape_count)

            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            plt.close(fig)

            return send_file(img, mimetype='image/png', as_attachment=True, download_name='plot.png')

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
