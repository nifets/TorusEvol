module TorusEvolBijectorsExt
    using TorusEvol
    using Bijectors
    using TorusEvol: DomainSets

    function Bijectors.bijector(d::Union{TorusEvol.WrappedNormal,
                                    TorusEvol.WrappedDiffusionTransition,
                                    TorusEvol.ResettingProcessTransition})
        Bijectors.Logit(DomainSets.infimum(domain(d)), DomainSets.supremum(domain(d)))
    end
end
