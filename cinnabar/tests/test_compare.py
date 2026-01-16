from cinnabar.compare import compare_and_rank_femaps
import networkx as nx
import numpy as np
from cinnabar import FEMap


def test_compare_and_rank_femaps(fe_map):
    graph1 = nx.MultiDiGraph()
    graph2 = nx.MultiDiGraph()
    for a, b, data in fe_map.to_networkx().edges(data=True):
        new_data = data.copy()
        if data["source"] != "reverse" and data["computational"]:
            # add noise to the result
            new_result = data["DG"] + np.random.normal(0, data["uncertainty"].m) * data["DG"].u
            new_data["DG"] = new_result
            graph1.add_edge(a, b, **new_data)
            # and add the reverse edge
            rev_data = new_data.copy()
            rev_data["source"] = "reverse"
            rev_data["DG"] = -new_data["DG"]
            graph1.add_edge(b, a, **rev_data)
            # add a large value to the second graph to simulate a bad prediction
            new_result2 = data["DG"] + 1.5 * data["DG"].u
            new_data["DG"] = new_result2
            graph2.add_edge(a, b, **new_data)
            # add the reverse edge
            rev_data2 = new_data.copy()
            rev_data2["source"] = "reverse"
            rev_data2["DG"] = -new_data["DG"]
            graph2.add_edge(b, a, **rev_data2)

        else:
            graph1.add_edge(a, b, **data)
            graph2.add_edge(a, b, **data)
    fe_map_2 = FEMap.from_networkx(graph1)
    fe_map_3 = FEMap.from_networkx(graph2)

    t1, t2 = compare_and_rank_femaps([fe_map, fe_map_2, fe_map_3], ["FE Map 1", "FE Map 2", "FE Map 3"], prediction_type="nodewise", rank_metric="rho")

    # check that FE Map 3 is ranked worst
    assert t1[t1["Model"] == "FE Map 3"]["CLD"].values[0] == "b"
    # check that FE Map 1 and FE Map 2 are ranked better
    assert t1[t1["Model"] == "FE Map 1"]["CLD"].values[0] == "a"
    assert t1[t1["Model"] == "FE Map 2"]["CLD"].values[0] == "a"
    # check that the comparison table has all three models and corrected p-values
    assert len(t2) == 3
    assert "p-value corrected" in t2.columns
