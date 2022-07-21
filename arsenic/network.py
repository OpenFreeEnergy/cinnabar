from pydantic import BaseModel, Extra
import networkx

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
    #TODO add units
    deltaG: float
    #TODO Use https://github.com/lebigot/uncertainties/
    variance: float


class RelativeResult(ExperimentalResult):
    pass



class FEMap():
    self._graph = networkx.DiGraph()

