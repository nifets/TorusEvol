# Helper functions
function ang_dist(ϕ_x, ϕ_y, ψ_x, ψ_y)
    d = sqrt(4 - 2*cos(ϕ_x - ϕ_y) - 2*cos(ψ_x - ψ_y))
    return d
end
