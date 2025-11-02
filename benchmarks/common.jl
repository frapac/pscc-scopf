using LazyArtifacts

include(joinpath(@__DIR__, "..", "data", "contingencies.jl"))

DATA_DIR = joinpath(artifact"ExaData", "ExaData")

CASES = [
    ("case118", 100),
    ("case300", 10),
    ("case_ACTIVSg200", 10),
    ("case_ACTIVSg200", 50),
    ("case_ACTIVSg200", 100),
    ("case_ACTIVSg500", 10),
    ("case_ACTIVSg500", 50),
    ("case_ACTIVSg500", 100),
    ("case1354pegase", 8),
    ("case1354pegase", 16),
    ("case1354pegase", 32),
    ("case_ACTIVSg2000", 8),
    ("case_ACTIVSg2000", 16),
    ("case2869pegase", 8),
    ("case2869pegase", 16),
]
