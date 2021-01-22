import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

import GrammarPredictionOpenDays
GrammarPredictionOpenDays.run_main()