# this will combine a single command that will run all analysis and save everything
from freeenergyframework import wrangle, plotting


def run_analysis(csv, prefix='out', target_name=''):
    raw_results = wrangle.read_csv(csv)
    network = wrangle.FEMap(raw_results)

    # this generates the three plots that we need
    network.draw_graph(filename=f'{prefix}-{target_name}-network.png',title=f'{target_name}')
    plotting.plot_DDGs(network.results, title=f'{prefix}-{target_name}', filename=f'{prefix}-{target_name}-DDGs.png')
    plotting.plot_DGs(network.graph, title=f'{prefix}-{target_name}', filename=f'{prefix}-{target_name}-DGs.png')


if __name__ == '__main__':
    import sys


    # TODO make this more flexible!!!
    csv = sys.argv[1]
    prefix = sys.argv[2]
    target_name = sys.argv[3]

    run_analysis(csv, prefix=prefix, target_name=target_name)
