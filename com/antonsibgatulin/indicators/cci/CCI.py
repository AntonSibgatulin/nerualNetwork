from com.antonsibgatulin.indicators.ma import MovingAverage


class CCI:
    def __init__(self, InpCCIPeriod):
        self.InpCCIPeriod = InpCCIPeriod
        self.ExtCCIBuffer = [];
        self.ExtPriceBuffer = []
        self.ExtMovBuffer = []
