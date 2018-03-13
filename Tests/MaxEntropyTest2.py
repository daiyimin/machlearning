from StatisticLearning import MaxEntroPyGIS
from Tests import TestUtils
import pandas as pd

x, y = TestUtils.load_csv("TestData/MaxEnt.data")
xdf = pd.DataFrame(x)
ys = pd.Series(y)

max_ent = MaxEntroPyGIS()
max_ent.train(xdf, ys)

xv = pd.Series(["Sunny"])
max_ent.predict(xv)