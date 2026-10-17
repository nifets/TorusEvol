module TorusEvolBijectorsExt
    using TorusEvol
    using Bijectors

    function Bijectors.bijector(d::Union{TorusEvol.WrappedNormal,
                                    TorusEvol.WrappedDiffusionTransition,
                                    TorusEvol.ResettingProcessTransition})
        lower, upper = TorusEvol.box(domain(d))
        Bijectors.Logit(lower, upper)
    end
end
