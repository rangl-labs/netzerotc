# -*- coding: utf-8 -*-
"""
#Created on Fri Oct  8 2021

@author: Administrator
"""

import numpy as np
import time
from pycel import ExcelCompiler

Pathways2Net0 = ExcelCompiler("Pathways to Net Zero - Simplified.xlsx")

Spreadsheets = np.array(['GALE!','CCUS!','Outputs!'])
ColumnInds_BySheets = np.array([np.array(['P','X','Y']), 
                                np.array(['O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI']), 
                                np.array(['O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI'])])
RowInds_BySheets = np.array([np.arange(35,36+20),
                             np.array([23,24,26,68]),
                             np.array([24,28,32,36,41,25,29,33,37,42,26,30,34,38,43,148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166])])

for iSheet in np.arange(len(Spreadsheets)):
    for iColumn in ColumnInds_BySheets[iSheet]:
        for iRow in RowInds_BySheets[iSheet]:
            Pathways2Net0.evaluate(Spreadsheets[iSheet] + iColumn + str(iRow))

Pathways2Net0.to_file("PathwaysToNetZero_Simplified_Compiled")


start = time.time()
Pathways2Net0_Loaded = ExcelCompiler.from_file("PathwaysToNetZero_Simplified_Compiled")
end = time.time()
print(f"INFO: took {end - start} seconds to load from serialised file")

