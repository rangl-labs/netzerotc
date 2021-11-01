# -*- coding: utf-8 -*-
"""
#Created on Fri Oct  8 2021

@author: Administrator
"""

import numpy as np
import time
from pycel import ExcelCompiler

# Pathways2Net0 = ExcelCompiler("Pathways to Net Zero - Simplified.xlsx")
Pathways2Net0 = ExcelCompiler("Pathways to Net Zero - Simplified - Anonymized.xlsx")

Spreadsheets = np.array(["GALE!", "CCUS!", "Outputs!"])

# fmt: off
ColumnInds_BySheets = np.array([np.array(['P','X','Y']), 
                                np.array(['O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI']), 
                                np.array(['O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI'])])
RowInds_BySheets = np.array([np.arange(35,36+20),
                             np.array([23,24,26,68]),
                             np.array([24,28,32,36,41, 25,29,33,37,42, 26,30,34,38,43, 67, 71, 
                                       148, 149, 150, 153, 154, 155, 158, 159, 163, 164, 165, 166])])
# fmt: on

for iSheet in np.arange(len(Spreadsheets)):
    for iColumn in ColumnInds_BySheets[iSheet]:
        for iRow in RowInds_BySheets[iSheet]:
            Pathways2Net0.evaluate(Spreadsheets[iSheet] + iColumn + str(iRow))

# Pathways2Net0.to_file("PathwaysToNetZero_Simplified_Compiled")
# Pathways2Net0.to_file("PathwaysToNetZero_Simplified_FullOutputs_Compiled")
Pathways2Net0.to_file("PathwaysToNetZero_Simplified_Anonymized_Compiled")


start = time.time()
# Pathways2Net0_Loaded = ExcelCompiler.from_file("PathwaysToNetZero_Simplified_Compiled")
# Pathways2Net0_Loaded = ExcelCompiler.from_file("PathwaysToNetZero_Simplified_FullOutputs_Compiled")
Pathways2Net0_Loaded = ExcelCompiler.from_file(
    "PathwaysToNetZero_Simplified_Anonymized_Compiled"
)
end = time.time()
print(f"INFO: took {end - start} seconds to load from serialised file")


## IEV model:
# IEVmodel = ExcelCompiler("OGTC-ORE Catapult IEV pathways economic model v01.06 FINAL - Original.xlsx")
IEVmodel = ExcelCompiler(
    "OGTC-ORE Catapult IEV pathways economic model v01.06 FINAL - Original Master sheet!C2=Gale.xlsx"
)

Spreadsheets = np.array(["Master sheet!", "Pathways inputs!"])
# fmt: off
ColumnInds_BySheets = np.array([np.array(['N','O','P','Q','R','S','T','U','V','W','X','Y', 'Z','AA','AB','AC','AD','AE','AF','AG','AH']), 
                                np.array(['O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AF','AG','AH','AI'])])
RowInds_BySheets = np.array([np.array([36,37,70]), 
                             np.array([24,28,32,36,41, 25,29,33,37,42, 26,30,34,38,43, 67, 71])])
# fmt: on

for iSheet in np.arange(len(Spreadsheets)):
    for iColumn in ColumnInds_BySheets[iSheet]:
        for iRow in RowInds_BySheets[iSheet]:
            IEVmodel.evaluate(Spreadsheets[iSheet] + iColumn + str(iRow))

# IEVmodel.to_file("IEV_pathways_economic_model_Compiled")
IEVmodel.to_file("IEV_pathways_economic_model_Master_sheet_C2SetToGale_Compiled")


start = time.time()
# IEVmodel_Loaded = ExcelCompiler.from_file("IEV_pathways_economic_model_Compiled")
IEVmodel_Loaded = ExcelCompiler.from_file(
    "IEV_pathways_economic_model_Master_sheet_C2SetToGale_Compiled"
)
end = time.time()
print(f"INFO: took {end - start} seconds to load from serialised file")
