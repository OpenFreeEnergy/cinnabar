class Result(object):
    def __init__(self,ligandA,ligandB,exp_DDG,exp_dDDG,calc_DDG,mbar_error,other_error):
        self.ligandA = str(ligandA)
        self.ligandB = str(ligandB)
        self.exp_DDG = float(exp_DDG)
        self.dexp_DDG = float(exp_dDDG)
        # scope for an experimental dDDG?
        self.calc_DDG = float(calc_DDG)
        self.mbar_dDDG = float(mbar_error)
        self.other_dDDG = float(other_error)
        self.dcalc_DDG = self.mbar_dDDG+self.other_dDDG # is this definitely always additive?


def read_csv(filename):
    raw_results = []
    with open(filename,'r') as f:
        for line in f:
            if line[0] != '#':
                raw_results.append(Result(*line.split(',')))
    return raw_results
