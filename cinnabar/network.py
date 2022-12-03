from pydantic import BaseModel, Extra
import networkx as nx


class Result(BaseModel):
    """
    Settings and modifications we want for all result classes.
    """

    class Config:
        extra = Extra.forbid
        # Immutability in python is never strict.
        # If developers are determined/stupid
        # they can always modify a so-called "immutable" object.
        # https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = False
        arbitrary_types_allowed = True
        smart_union = True


class ExperimentalResult(Result):
    # TODO add units
    deltaG: float
    # TODO Use https://github.com/lebigot/uncertainties/
    variance: float


class RelativeResult(ExperimentalResult):
    pass


class FEMap:
    self._graph = nx.DiGraph()

    def add_node(self, ligand_name: str, experimental_result: ExperimentalResult):
        self._graph.add_node(ligand_name, experimental_result=experimental_result)

    def add_edge(self, ligand_A: str, ligand_B: str, relative_result: RelativeResult):
        self._graph.add_edge(ligand_A, ligand_B, relative_result=relative_result)

    def save(self, file_name):
        # TODO will need to loop over the graph, and make a new graph to save
        # since it doesn't look like graphml makes it easy to define how objects should
        # be seralized so we will need to do that ourselves
        nx.write_graphml(self._graph, filename)
