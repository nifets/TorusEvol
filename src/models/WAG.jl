using DelimitedFiles

const WAG_DIR = pkgdir(@__MODULE__, "data", "params")

const WAG_S = LowerTriangular(map((x) -> x=="" ? 0.0 : x,
                        readdlm(joinpath(WAG_DIR, "WAG_S.csv"); skipblanks=false, dims=(20,20))))

const WAG_Π = normalize!(vec(readdlm(joinpath(WAG_DIR, "WAG_PI.csv"))), 1)

WAG_substitution_process(Π=WAG_Π; r=1.0) = substitution_process(WAG_S, Π; r=r);
