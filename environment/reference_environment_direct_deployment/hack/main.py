#!/usr/bin/env python

import time
from pycel import ExcelCompiler

# Simple test

cell = "Sheet1!B5"
test_excel = ExcelCompiler(filename="input.xlsx")
test_value = test_excel.evaluate(cell)
test_excel.to_file("test")
test_loaded = ExcelCompiler.from_file("test")

assert test_loaded.evaluate(cell) == test_excel.evaluate(cell)

# Production version

cell = "'Vision 2035 Production'!C4"
SOURCE = "../../compiled_workbook_objects/Pathways to Net Zero - Simplified.xlsx"

start = time.time()
production_excel = ExcelCompiler(filename=SOURCE)
end = time.time()
print(f"INFO: took {end - start} seconds to load natively")

production_value = production_excel.evaluate(cell)
production_excel.to_file("production")

start = time.time()
production_loaded = ExcelCompiler.from_file("production")
end = time.time()
print(f"INFO: took {end - start} seconds to load from serialised file")

assert production_loaded.evaluate(cell) == production_excel.evaluate(cell)
