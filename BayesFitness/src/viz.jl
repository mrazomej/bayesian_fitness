using Measures, CairoMakie
import ColorSchemes
import MCMCChains

"""
    `pboc_plotlyjs!()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `plotly` backend.
"""
function pboc_plotlyjs!()
    plotlyjs(
        background_color="#E3DCD0",
        background_color_outside="white",
        foreground_color_grid="#ffffff",
        gridlinewidth=0.5,
        guidefontfamily="Lucida Sans Unicode",
        guidefontsize=8,
        tickfontfamily="Lucida Sans Unicode",
        tickfontsize=7,
        titlefontfamily="Lucida Sans Unicode",
        titlefontsize=8,
        dpi=300,
        linewidth=1.25,
        legendtitlefontsize=8,
        legendfontsize=8,
        legend=(0.8, 0.8),
        foreground_color_legend="#E3DCD0",
        color_palette=:seaborn_colorblind,
        label=:none,
        fmt=:png
    )
end

"""
    `pboc_pyplot()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `pyplot` backend.
"""
function pboc_pyplot!()
    pyplot(
        background_color="#E3DCD0",
        background_color_outside="white",
        foreground_color_grid="#ffffff",
        guidefontfamily="Lucida Sans Unicode",
        guidefontsize=8,
        tickfontfamily="Lucida Sans Unicode",
        tickfontsize=7,
        titlefontfamily="Lucida Sans Unicode",
        titlefontsize=8,
        linewidth=1.25,
        legendtitlefontsize=8,
        legendfontsize=8,
        legend=(0.8, 0.8),
        foreground_color_legend="#E3DCD0",
        color_palette=:seaborn_colorblind,
        label=:none,
        grid=true,
        gridcolor="white",
        gridlinewidth=1.5
    )
end

"""
    `pboc_gr()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `gr` backend.
"""
function pboc_gr!()
    gr(
        background_color="#E3DCD0",
        background_color_outside="white",
        guidefontfamily="Lucida Sans Unicode",
        guidefontsize=8,
        tickfontfamily="Lucida Sans Unicode",
        tickfontsize=7,
        titlefontfamily="Lucida Sans Unicode",
        titlefontsize=10,
        linewidth=1.25,
        legendtitlefontsize=8,
        legendfontsize=8,
        legend=:topright,
        background_color_legend="#E3DCD0",
        color_palette=:seaborn_colorblind,
        label=:none,
        grid=true,
        gridcolor="white",
        minorgridcolor="white",
        gridlinewidth=1.5,
        minorgridlinewidth=1.5,
        bottom_margin=5mm
    )
end

"""
    `pboc_makie()`

Set plotting default to that used in Physical Biology of the Cell, 2nd edition
for the `makie` plotting library. This can be for either the GLMakie or the
CairoMakie backends
"""
function pboc_makie!()
    # if ~isfile(assetpath("fonts", "Lucida-sans-Unicode-Regular.ttf"))
    #     @warn "Lucida sans Unicode Regular font not added to Makie Fonts. Add to `~/.julia/packages/Makie/gQOQF/assets/fonts/`. Defaulting to NotoSans."
    #     Font = assetpath("fonts", "NotoSans-Regular.tff")
    # else
    #     Font = assetpath("fonts", "Lucida-Sans-Unicode-Regular.ttf")
    # end

    Font = "Lucida Sans Regular"
    # Seaborn colorblind
    colors = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#d55e00",
        "#cc78bc",
        "#ca9161",
        "#fbafe4",
        "#949494",
        "#ece133",
        "#56b4e9"
    ]

    theme = Theme(
        Axis=(
            backgroundcolor="#E3DCD0",

            # Font sizes
            titlesize=16,
            xlabelsize=16,
            ylabelsize=16,
            xticklabelsize=14,
            yticklabelsize=14,

            # Font styles
            titlefont=Font,
            xticklabelfont=Font,
            yticklabelfont=Font,
            xlabelfont=Font,
            ylabelfont=Font,

            # Grid
            xgridwidth=1.25,
            ygridwidth=1.25,
            xgridcolor="white",
            ygridcolor="white",
            xminorgridcolor="white",
            yminorgridcolor="white",
            xminorgridvisible=false,
            xminorgridwidth=1.0,
            yminorgridvisible=false,
            yminorgridwidth=1.0,

            # Axis ticks
            minorticks=false,
            xticksvisible=false,
            yticksvisible=false,

            # Box
            rightspinevisible=false,
            leftspinevisible=false,
            topspinevisible=false,
            bottomspinevisible=false,
        ),
        Legend=(
            titlesize=15,
            labelsize=15,
            bgcolor="#E3DCD0",
        ),
        backgroundcolor="white",
        linewidth=1.25,
    )
    set_theme!(theme)
end

@doc raw"""
    `mcmc_trace_density!(fig::Figure, chain::MCMCChains.Chains; colors, labels)`
Function to plot the traces and density estimates side-to-side for each of the
parametres in the `MCMCChains.Chains` object.
# Arguments
- `fig::Figure`: Figure object to be populated with plot. This allows the user
  to decide the size of the figure outside of this function.
- `chain::MCMCChains.Chains`: Samples from the MCMC run generated with
  Turing.jl.
## Optional arguments
- `colors`: List of colors to be used in plot.
- `labels`: List of labels for each of the parameters. If not given, the default
  will be to use the names stored in the MCMCChains.Chains object
- `alpha::AbstractFloat=1`: Level of transparency for plots.
"""
function mcmc_trace_density!(
    fig::Figure,
    chain::MCMCChains.Chains;
    colors=ColorSchemes.seaborn_colorblind,
    labels=[],
    alpha::AbstractFloat=1.0
)
    # Extract parameters
    params = names(chain, :parameters)
    # Extract number of chains
    n_chains = length(MCMCChains.chains(chain))
    # Extract number of parameters
    n_samples = length(chain)

    # Check that the number of given labels is correct
    if (length(labels) > 0) & (length(labels) != length(params))
        error("The number of lables must match number of parameters")
    end # if

    # Check that the number of given colors is correct
    if length(colors) < n_chains
        error("Please give at least as many colors as chains in the MCMC")
    end # if

    # Loop through parameters
    for (i, param) in enumerate(params)
        # Check if labels were given
        if length(labels) > 0
            lab = labels[i]
        else
            lab = string(param)
        end # if
        # Add axis for chain iteration
        ax_trace = Axis(fig[i, 1]; ylabel=lab)
        # Inititalize axis for density plot
        ax_density = Axis(fig[i, 2]; ylabel=lab)
        # Loop through chains
        for chn in 1:n_chains
            # Extract values
            values = chain[:, param, chn]
            # Plot traces of walker
            lines!(ax_trace, 1:n_samples, values, color=(colors[chn], alpha))
            # Plot density
            density!(ax_density, values, color=(colors[chn], alpha))
        end # for

        # Hide y-axis decorations
        hideydecorations!(ax_trace; label=false)
        hideydecorations!(ax_density; label=false)

        # Check if it is bottom plot
        if i < length(params)
            # hide x-axis decoratiosn
            hidexdecorations!(ax_trace; grid=false)
        else
            # add x-label
            ax_trace.xlabel = "iteration"
            ax_density.xlabel = "parameter estimate"
        end # if
    end # for
end # function