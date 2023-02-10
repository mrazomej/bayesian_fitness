using Measures, CairoMakie
import ColorSchemes

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
